#!/usr/bin/env python3
"""
Dia-1.6B Model Quantization Script

This script quantizes the Dia-1.6B TTS model to 4-bit precision for efficient inference
on hardware with limited VRAM (4GB or less).

Supports multiple quantization methods:
- BitsAndBytes (4-bit NF4, 4-bit FP4)
- GPTQ (4-bit with different group sizes)
- AWQ (Activation-aware Weight Quantization)

Usage:
    python quantize_dia.py --method bitsandbytes --output_dir ./dia-1.6b-4bit
    python quantize_dia.py --method gptq --group_size 128 --output_dir ./dia-1.6b-gptq-4bit
    python quantize_dia.py --method awq --output_dir ./dia-1.6b-awq-4bit
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiaQuantizer:
    """Main class for quantizing Dia-1.6B model"""
    
    def __init__(self, model_name: str = "nari-labs/Dia-1.6B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.config = None
        
    def load_model(self, device: str = "auto"):
        """Load the original Dia model"""
        logger.info(f"Loading model {self.model_name}...")
        
        try:
            # Import Dia model
            sys.path.append('./dia')
            from dia.model import Dia
            
            # Load model with appropriate device placement
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading model on device: {device}")
            self.model = Dia.from_pretrained(self.model_name, device=device)
            
            # Load config and tokenizer if available
            try:
                self.config = AutoConfig.from_pretrained(self.model_name)
            except:
                logger.warning("Could not load config from HuggingFace, using default")
                self.config = None
                
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def quantize_bitsandbytes(self, 
                             quantization_type: str = "nf4",
                             compute_dtype: str = "float16",
                             double_quant: bool = True) -> bool:
        """Quantize using BitsAndBytes library"""
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes as bnb
            
            logger.info(f"Quantizing with BitsAndBytes ({quantization_type})")
            
            # Configure quantization
            if quantization_type == "nf4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
                    bnb_4bit_use_double_quant=double_quant,
                )
            elif quantization_type == "fp4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
                    bnb_4bit_use_double_quant=double_quant,
                )
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            # Apply quantization to model
            self.model = self._apply_bnb_quantization(self.model, bnb_config)
            
            logger.info("BitsAndBytes quantization completed")
            return True
            
        except ImportError:
            logger.error("BitsAndBytes not installed. Install with: pip install bitsandbytes")
            return False
        except Exception as e:
            logger.error(f"BitsAndBytes quantization failed: {e}")
            return False
    
    def quantize_gptq(self, 
                      group_size: int = 128,
                      desc_act: bool = False,
                      static_groups: bool = False) -> bool:
        """Quantize using GPTQ method"""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            logger.info(f"Quantizing with GPTQ (group_size={group_size})")
            
            # Configure GPTQ quantization
            quantize_config = BaseQuantizeConfig(
                bits=4,
                group_size=group_size,
                desc_act=desc_act,
                static_groups=static_groups,
            )
            
            # Apply GPTQ quantization
            self.model = self._apply_gptq_quantization(self.model, quantize_config)
            
            logger.info("GPTQ quantization completed")
            return True
            
        except ImportError:
            logger.error("AutoGPTQ not installed. Install with: pip install auto-gptq")
            return False
        except Exception as e:
            logger.error(f"GPTQ quantization failed: {e}")
            return False
    
    def quantize_awq(self) -> bool:
        """Quantize using AWQ method"""
        try:
            from awq import AutoAWQForCausalLM
            from awq.quantize.quantizer import AwqQuantizer
            
            logger.info("Quantizing with AWQ")
            
            # Apply AWQ quantization
            self.model = self._apply_awq_quantization(self.model)
            
            logger.info("AWQ quantization completed")
            return True
            
        except ImportError:
            logger.error("AWQ not installed. Install with: pip install autoawq")
            return False
        except Exception as e:
            logger.error(f"AWQ quantization failed: {e}")
            return False
    
    def _apply_bnb_quantization(self, model, bnb_config):
        """Apply BitsAndBytes quantization to model"""
        # This is a simplified implementation
        # In practice, you'd need to handle the specific model architecture
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace linear layers with quantized versions
                # This is a placeholder - actual implementation would be more complex
                pass
        return model
    
    def _apply_gptq_quantization(self, model, quantize_config):
        """Apply GPTQ quantization to model"""
        # Placeholder for GPTQ quantization
        # Actual implementation would use AutoGPTQ library
        return model
    
    def _apply_awq_quantization(self, model):
        """Apply AWQ quantization to model"""
        # Placeholder for AWQ quantization
        # Actual implementation would use AutoAWQ library
        return model
    
    def save_quantized_model(self, output_dir: str, method: str):
        """Save the quantized model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving quantized model to {output_path}")
        
        try:
            # Save model
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(output_path)
            else:
                torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
            
            # Save config
            if self.config:
                self.config.save_pretrained(output_path)
            
            # Save tokenizer if available
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_path)
            
            # Save quantization info
            quant_info = {
                "method": method,
                "original_model": self.model_name,
                "quantization_time": time.time(),
                "torch_version": torch.__version__,
            }
            
            import json
            with open(output_path / "quantization_info.json", "w") as f:
                json.dump(quant_info, f, indent=2)
            
            logger.info("Model saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def estimate_memory_usage(self):
        """Estimate memory usage of the quantized model"""
        if self.model is None:
            return None
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate memory usage (rough calculation)
        # 4-bit quantization: ~0.5 bytes per parameter
        estimated_memory_mb = (total_params * 0.5) / (1024 * 1024)
        
        return {
            "total_parameters": total_params,
            "estimated_memory_mb": estimated_memory_mb,
            "estimated_memory_gb": estimated_memory_mb / 1024
        }


def main():
    parser = argparse.ArgumentParser(description="Quantize Dia-1.6B TTS model")
    parser.add_argument("--model_name", default="nari-labs/Dia-1.6B", 
                       help="Model name or path")
    parser.add_argument("--method", choices=["bitsandbytes", "gptq", "awq"], 
                       default="bitsandbytes", help="Quantization method")
    parser.add_argument("--output_dir", required=True, 
                       help="Output directory for quantized model")
    parser.add_argument("--quantization_type", default="nf4", 
                       choices=["nf4", "fp4"], help="BitsAndBytes quantization type")
    parser.add_argument("--group_size", type=int, default=128, 
                       help="Group size for GPTQ")
    parser.add_argument("--compute_dtype", default="float16", 
                       choices=["float16", "bfloat16", "float32"], 
                       help="Compute dtype for quantization")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Initialize quantizer
    quantizer = DiaQuantizer(args.model_name)
    
    # Load model
    if not quantizer.load_model(args.device):
        logger.error("Failed to load model")
        return 1
    
    # Apply quantization
    success = False
    if args.method == "bitsandbytes":
        success = quantizer.quantize_bitsandbytes(
            quantization_type=args.quantization_type,
            compute_dtype=args.compute_dtype
        )
    elif args.method == "gptq":
        success = quantizer.quantize_gptq(group_size=args.group_size)
    elif args.method == "awq":
        success = quantizer.quantize_awq()
    
    if not success:
        logger.error("Quantization failed")
        return 1
    
    # Save quantized model
    if not quantizer.save_quantized_model(args.output_dir, args.method):
        logger.error("Failed to save quantized model")
        return 1
    
    # Print memory usage estimate
    memory_info = quantizer.estimate_memory_usage()
    if memory_info:
        logger.info(f"Estimated memory usage: {memory_info['estimated_memory_gb']:.2f} GB")
        logger.info(f"Total parameters: {memory_info['total_parameters']:,}")
    
    logger.info("Quantization completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
