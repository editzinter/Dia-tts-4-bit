#!/usr/bin/env python3
"""
Preset-based quantization script for Dia-1.6B model

This script uses predefined presets for different hardware configurations
to make quantization easier for users.

Usage:
    python preset_quantize.py --preset ultra_low_vram --output ./quantized_model
    python preset_quantize.py --preset balanced --output ./quantized_model
    python preset_quantize.py --list-presets
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add current directory to path
sys.path.append('.')
sys.path.append('./dia')

from simple_quantize import SimpleDiaQuantizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PresetQuantizer:
    """Quantizer that uses predefined presets"""
    
    def __init__(self, config_path: str = "configs/quantization_presets.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load quantization presets configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return {}
    
    def list_presets(self):
        """List all available presets"""
        presets = self.config.get("presets", {})
        
        print("\n🎛️  Available Quantization Presets")
        print("=" * 50)
        
        for name, preset in presets.items():
            print(f"\n📋 {name}")
            print(f"   Description: {preset.get('description', 'No description')}")
            print(f"   Method: {preset.get('method', 'Unknown')}")
            print(f"   Expected VRAM: {preset.get('expected_vram_mb', 'Unknown')} MB")
            print(f"   Expected Quality: {preset.get('expected_quality', 'Unknown')}")
            print(f"   Use Case: {preset.get('use_case', 'General')}")
        
        # Hardware recommendations
        hardware_recs = self.config.get("hardware_recommendations", {})
        if hardware_recs:
            print("\n🖥️  Hardware Recommendations")
            print("=" * 50)
            
            for hardware, rec in hardware_recs.items():
                print(f"\n💻 {hardware}")
                print(f"   Recommended: {rec.get('recommended_preset', 'None')}")
                print(f"   Alternatives: {', '.join(rec.get('alternative_presets', []))}")
                if rec.get('notes'):
                    print(f"   Notes: {rec['notes']}")
    
    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get a specific preset configuration"""
        presets = self.config.get("presets", {})
        
        if preset_name not in presets:
            available = list(presets.keys())
            raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")
        
        return presets[preset_name]
    
    def quantize_with_preset(self, preset_name: str, output_dir: str, 
                           model_name: str = "nari-labs/Dia-1.6B", 
                           device: str = "auto") -> bool:
        """Quantize model using a preset"""
        
        logger.info(f"🎛️  Using preset: {preset_name}")
        
        # Get preset configuration
        preset = self.get_preset(preset_name)
        method = preset.get("method", "4bit")
        parameters = preset.get("parameters", {})
        
        logger.info(f"📋 Preset details:")
        logger.info(f"   Description: {preset.get('description', 'No description')}")
        logger.info(f"   Method: {method}")
        logger.info(f"   Expected VRAM: {preset.get('expected_vram_mb', 'Unknown')} MB")
        logger.info(f"   Expected Quality: {preset.get('expected_quality', 'Unknown')}")
        
        # Initialize quantizer based on method
        if method in ["4bit", "dynamic"]:
            return self._quantize_simple(method, output_dir, model_name, device, parameters)
        elif method in ["bitsandbytes", "gptq", "awq"]:
            return self._quantize_advanced(method, output_dir, model_name, device, parameters)
        else:
            logger.error(f"Unknown quantization method: {method}")
            return False
    
    def _quantize_simple(self, method: str, output_dir: str, model_name: str, 
                        device: str, parameters: Dict[str, Any]) -> bool:
        """Quantize using simple quantization methods"""
        
        try:
            # Initialize simple quantizer
            quantizer = SimpleDiaQuantizer(model_name)
            
            # Load model
            if not quantizer.load_model(device):
                return False
            
            # Apply quantization
            if method == "4bit":
                success = quantizer.quantize_linear_layers()
            elif method == "dynamic":
                success = quantizer.apply_dynamic_quantization()
            else:
                logger.error(f"Unsupported simple method: {method}")
                return False
            
            if not success:
                return False
            
            # Save model
            return quantizer.save_quantized_model(output_dir, f"preset_{method}")
            
        except Exception as e:
            logger.error(f"Simple quantization failed: {e}")
            return False
    
    def _quantize_advanced(self, method: str, output_dir: str, model_name: str, 
                          device: str, parameters: Dict[str, Any]) -> bool:
        """Quantize using advanced quantization methods"""
        
        logger.warning(f"Advanced method '{method}' requires additional dependencies")
        logger.info("Install with: pip install bitsandbytes auto-gptq optimum")
        
        try:
            # Try to import and use the advanced quantizer
            from quantize_dia import DiaQuantizer
            
            quantizer = DiaQuantizer(model_name)
            
            if not quantizer.load_model(device):
                return False
            
            # Apply quantization based on method
            success = False
            if method == "bitsandbytes":
                success = quantizer.quantize_bitsandbytes(
                    quantization_type=parameters.get("quantization_type", "nf4"),
                    compute_dtype=parameters.get("compute_dtype", "float16"),
                    double_quant=parameters.get("double_quant", True)
                )
            elif method == "gptq":
                success = quantizer.quantize_gptq(
                    group_size=parameters.get("group_size", 128),
                    desc_act=parameters.get("desc_act", False),
                    static_groups=parameters.get("static_groups", False)
                )
            elif method == "awq":
                success = quantizer.quantize_awq()
            
            if not success:
                return False
            
            return quantizer.save_quantized_model(output_dir, f"preset_{method}")
            
        except ImportError:
            logger.error(f"Advanced quantization libraries not available")
            logger.info("Falling back to simple 4-bit quantization...")
            return self._quantize_simple("4bit", output_dir, model_name, device, {})
        except Exception as e:
            logger.error(f"Advanced quantization failed: {e}")
            return False
    
    def recommend_preset(self, vram_gb: float = None, use_case: str = None) -> str:
        """Recommend a preset based on hardware and use case"""
        
        if vram_gb is not None:
            if vram_gb <= 4:
                return "ultra_low_vram"
            elif vram_gb <= 6:
                return "balanced"
            elif vram_gb <= 8:
                return "high_quality"
            else:
                return "high_quality"
        
        if use_case:
            use_case = use_case.lower()
            if "mobile" in use_case or "low" in use_case:
                return "ultra_low_vram"
            elif "cpu" in use_case:
                return "cpu_optimized"
            elif "quality" in use_case:
                return "high_quality"
        
        # Default recommendation
        return "balanced"


def main():
    parser = argparse.ArgumentParser(description="Preset-based Dia model quantization")
    
    # Main actions
    parser.add_argument("--list-presets", action="store_true", 
                       help="List all available presets")
    parser.add_argument("--preset", help="Preset name to use")
    parser.add_argument("--recommend", action="store_true", 
                       help="Get preset recommendation")
    
    # Configuration
    parser.add_argument("--output", help="Output directory for quantized model")
    parser.add_argument("--model", default="nari-labs/Dia-1.6B", 
                       help="Model name or path")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--config", default="configs/quantization_presets.json",
                       help="Path to presets configuration file")
    
    # For recommendations
    parser.add_argument("--vram-gb", type=float, 
                       help="Available VRAM in GB for recommendation")
    parser.add_argument("--use-case", 
                       help="Use case for recommendation (mobile, cpu, quality, etc.)")
    
    args = parser.parse_args()
    
    # Initialize preset quantizer
    preset_quantizer = PresetQuantizer(args.config)
    
    # Handle different actions
    if args.list_presets:
        preset_quantizer.list_presets()
        return 0
    
    if args.recommend:
        recommended = preset_quantizer.recommend_preset(args.vram_gb, args.use_case)
        print(f"\n🎯 Recommended preset: {recommended}")
        
        # Show preset details
        try:
            preset = preset_quantizer.get_preset(recommended)
            print(f"   Description: {preset.get('description', 'No description')}")
            print(f"   Expected VRAM: {preset.get('expected_vram_mb', 'Unknown')} MB")
            print(f"   Expected Quality: {preset.get('expected_quality', 'Unknown')}")
        except ValueError:
            pass
        
        return 0
    
    if args.preset:
        if not args.output:
            logger.error("--output is required when using --preset")
            return 1
        
        try:
            success = preset_quantizer.quantize_with_preset(
                args.preset, args.output, args.model, args.device
            )
            
            if success:
                logger.info("🎉 Quantization completed successfully!")
                return 0
            else:
                logger.error("❌ Quantization failed")
                return 1
                
        except ValueError as e:
            logger.error(f"❌ {e}")
            return 1
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return 1
    
    # If no action specified, show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
