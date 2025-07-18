#!/usr/bin/env python3
"""
Simple Dia Model Quantization Script

This script provides a simplified approach to quantize the Dia-1.6B model
using PyTorch's built-in quantization capabilities and custom 4-bit quantization.

This approach doesn't require additional quantization libraries and works
with the existing Dia model architecture.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic
import numpy as np

# Add dia to path
sys.path.append('./dia')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Simple4BitLinear(nn.Module):
    """Simple 4-bit quantized linear layer"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weights as int8 (2 weights per byte)
        self.register_buffer('weight_quantized', torch.zeros((out_features, (in_features + 1) // 2), dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.zeros(out_features, dtype=torch.float32))
        self.register_buffer('weight_zero_point', torch.zeros(out_features, dtype=torch.float32))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None
    
    @classmethod
    def from_linear(cls, linear_layer: nn.Linear):
        """Convert a regular linear layer to 4-bit quantized version"""
        quantized = cls(linear_layer.in_features, linear_layer.out_features, 
                       bias=linear_layer.bias is not None)
        
        # Quantize weights to 4-bit
        weight = linear_layer.weight.data.float()
        quantized_weight, scale, zero_point = quantize_weights_4bit(weight)
        
        quantized.weight_quantized.copy_(quantized_weight)
        quantized.weight_scale.copy_(scale)
        quantized.weight_zero_point.copy_(zero_point)
        
        if linear_layer.bias is not None:
            quantized.bias.copy_(linear_layer.bias.data)
        
        return quantized
    
    def forward(self, x):
        # Dequantize weights
        weight = dequantize_weights_4bit(self.weight_quantized, self.weight_scale, self.weight_zero_point)
        return F.linear(x, weight, self.bias)


def quantize_weights_4bit(weights: torch.Tensor):
    """Quantize weights to 4-bit representation"""
    # Calculate per-channel quantization parameters
    w_min = weights.min(dim=1, keepdim=True)[0]
    w_max = weights.max(dim=1, keepdim=True)[0]
    
    # Avoid division by zero
    scale = (w_max - w_min) / 15.0  # 4-bit has 16 levels (0-15)
    scale = torch.clamp(scale, min=1e-8)
    zero_point = w_min
    
    # Quantize to 4-bit
    quantized = torch.round((weights - zero_point) / scale).clamp(0, 15)
    
    # Pack two 4-bit values into one uint8
    quantized = quantized.to(torch.uint8)
    out_features, in_features = quantized.shape
    packed_features = (in_features + 1) // 2
    
    packed = torch.zeros((out_features, packed_features), dtype=torch.uint8, device=weights.device)
    
    for i in range(0, in_features, 2):
        if i + 1 < in_features:
            # Pack two 4-bit values
            packed[:, i // 2] = quantized[:, i] | (quantized[:, i + 1] << 4)
        else:
            # Handle odd number of features
            packed[:, i // 2] = quantized[:, i]
    
    return packed, scale.squeeze(), zero_point.squeeze()


def dequantize_weights_4bit(packed_weights: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """Dequantize 4-bit weights back to float"""
    out_features, packed_features = packed_weights.shape
    in_features = packed_features * 2
    
    # Unpack 4-bit values
    unpacked = torch.zeros((out_features, in_features), dtype=torch.float32, device=packed_weights.device)
    
    for i in range(packed_features):
        # Extract lower 4 bits
        unpacked[:, i * 2] = (packed_weights[:, i] & 0x0F).float()
        # Extract upper 4 bits
        if i * 2 + 1 < in_features:
            unpacked[:, i * 2 + 1] = ((packed_weights[:, i] >> 4) & 0x0F).float()
    
    # Dequantize
    scale = scale.unsqueeze(1)
    zero_point = zero_point.unsqueeze(1)
    dequantized = unpacked * scale + zero_point
    
    return dequantized


class SimpleDiaQuantizer:
    """Simple quantizer for Dia model"""
    
    def __init__(self, model_name: str = "nari-labs/Dia-1.6B"):
        self.model_name = model_name
        self.model = None
        self.original_size = 0
        self.quantized_size = 0
    
    def load_model(self, device: str = "cpu"):
        """Load the Dia model"""
        try:
            from dia.model import Dia
            
            logger.info(f"Loading Dia model: {self.model_name}")
            
            # Load with CPU to avoid VRAM issues in VM
            self.model = Dia.from_pretrained(self.model_name, device=device)
            
            # Calculate original model size
            self.original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            logger.info(f"Original model size: {self.original_size / 1024**3:.2f} GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def quantize_linear_layers(self):
        """Quantize all linear layers in the model to 4-bit"""
        logger.info("Quantizing linear layers to 4-bit...")
        
        quantized_layers = 0
        
        def quantize_module(module, name=""):
            nonlocal quantized_layers
            
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child, nn.Linear):
                    # Replace with quantized version
                    quantized_layer = Simple4BitLinear.from_linear(child)
                    setattr(module, child_name, quantized_layer)
                    quantized_layers += 1
                    logger.debug(f"Quantized layer: {full_name}")
                else:
                    # Recursively quantize child modules
                    quantize_module(child, full_name)
        
        quantize_module(self.model)
        
        logger.info(f"Quantized {quantized_layers} linear layers")
        
        # Calculate quantized model size (approximate)
        self.quantized_size = self.original_size * 0.25  # 4-bit is roughly 1/4 of 32-bit
        logger.info(f"Estimated quantized size: {self.quantized_size / 1024**3:.2f} GB")
        
        return quantized_layers > 0
    
    def apply_dynamic_quantization(self):
        """Apply PyTorch's dynamic quantization"""
        logger.info("Applying dynamic quantization...")
        
        try:
            # Apply dynamic quantization to linear layers
            self.model = quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=torch.qint8
            )
            
            logger.info("Dynamic quantization applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            return False
    
    def save_quantized_model(self, output_dir: str, method: str):
        """Save the quantized model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving quantized model to {output_path}")
        
        try:
            # Save model state dict
            torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
            
            # Save model architecture (if possible)
            try:
                torch.save(self.model, output_path / "model.pth")
            except Exception as e:
                logger.warning(f"Could not save full model: {e}")
            
            # Save quantization metadata
            metadata = {
                "method": method,
                "original_model": self.model_name,
                "quantization_time": time.time(),
                "original_size_gb": self.original_size / 1024**3,
                "quantized_size_gb": self.quantized_size / 1024**3,
                "compression_ratio": self.original_size / self.quantized_size if self.quantized_size > 0 else 1.0,
                "torch_version": torch.__version__,
                "python_version": sys.version,
            }
            
            with open(output_path / "quantization_info.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Create a simple loading script
            loading_script = '''#!/usr/bin/env python3
"""
Load quantized Dia model

Usage:
    python load_quantized.py --model_path ./quantized_model
"""

import torch
import sys
import argparse
from pathlib import Path

# Add dia to path
sys.path.append('./dia')

def load_quantized_dia(model_path: str, device: str = "cpu"):
    """Load quantized Dia model"""
    model_path = Path(model_path)
    
    try:
        # Try to load full model first
        if (model_path / "model.pth").exists():
            model = torch.load(model_path / "model.pth", map_location=device)
            print("Loaded full quantized model")
            return model
        
        # Fallback to loading state dict
        elif (model_path / "pytorch_model.bin").exists():
            from dia.model import Dia
            
            # Load original model structure
            model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
            
            # Load quantized weights
            state_dict = torch.load(model_path / "pytorch_model.bin", map_location=device)
            model.load_state_dict(state_dict)
            
            print("Loaded quantized model from state dict")
            return model
        
        else:
            raise FileNotFoundError("No model files found")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to quantized model")
    parser.add_argument("--device", default="cpu", help="Device to load model on")
    
    args = parser.parse_args()
    
    model = load_quantized_dia(args.model_path, args.device)
    if model:
        print("Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
    else:
        print("Failed to load model")
'''
            
            with open(output_path / "load_quantized.py", "w") as f:
                f.write(loading_script)
            
            # Make loading script executable
            os.chmod(output_path / "load_quantized.py", 0o755)
            
            logger.info("Quantized model saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Simple Dia model quantization")
    parser.add_argument("--model_name", default="nari-labs/Dia-1.6B", help="Model name")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--method", choices=["4bit", "dynamic"], default="4bit", 
                       help="Quantization method")
    parser.add_argument("--device", default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize quantizer
    quantizer = SimpleDiaQuantizer(args.model_name)
    
    # Load model
    if not quantizer.load_model(args.device):
        return 1
    
    # Apply quantization
    success = False
    if args.method == "4bit":
        success = quantizer.quantize_linear_layers()
    elif args.method == "dynamic":
        success = quantizer.apply_dynamic_quantization()
    
    if not success:
        logger.error("Quantization failed")
        return 1
    
    # Save quantized model
    if not quantizer.save_quantized_model(args.output_dir, args.method):
        return 1
    
    logger.info("Quantization completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
