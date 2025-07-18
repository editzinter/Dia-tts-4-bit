#!/usr/bin/env python3
"""
Example script for generating speech with quantized Dia model

Usage:
    python generate_speech.py --text "Hello world" --model ./quantized_models/dia-1.6b-4bit
    python generate_speech.py --text "Hello world" --original  # Use original model
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append('..')
sys.path.append('../dia')

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str = None, use_original: bool = False, device: str = "auto"):
    """Load either quantized or original model"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if use_original:
        logger.info("Loading original Dia-1.6B model...")
        from dia.model import Dia
        model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
        return model, "Original Dia-1.6B"
    
    elif model_path:
        logger.info(f"Loading quantized model from {model_path}...")
        model_path = Path(model_path)
        
        # Try different loading methods
        if (model_path / "model.pth").exists():
            model = torch.load(model_path / "model.pth", map_location=device)
            return model, f"Quantized ({model_path.name})"
        
        elif (model_path / "pytorch_model.bin").exists():
            from dia.model import Dia
            model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
            state_dict = torch.load(model_path / "pytorch_model.bin", map_location=device)
            model.load_state_dict(state_dict)
            return model, f"Quantized ({model_path.name})"
        
        else:
            raise FileNotFoundError(f"No model files found in {model_path}")
    
    else:
        raise ValueError("Must specify either --model or --original")


def generate_speech(model, text: str, output_file: str = "output.mp3", **kwargs):
    """Generate speech from text"""
    
    logger.info(f"Generating speech for: '{text}'")
    logger.info(f"Output file: {output_file}")
    
    # Default generation parameters
    generation_params = {
        "max_tokens": 512,
        "verbose": True,
        "use_torch_compile": False,
        "cfg_scale": 3.0,
        "temperature": 1.2,
        "top_p": 0.95,
    }
    
    # Update with user parameters
    generation_params.update(kwargs)
    
    try:
        # Generate audio
        import time
        start_time = time.time()
        
        audio = model.generate(text, **generation_params)
        
        generation_time = time.time() - start_time
        
        if audio is not None:
            # Save audio
            model.save_audio(output_file, audio)
            
            # Calculate metrics
            audio_duration = len(audio) / 44100  # Assuming 44.1kHz
            real_time_factor = audio_duration / generation_time
            
            logger.info(f"✅ Generation successful!")
            logger.info(f"   Audio duration: {audio_duration:.2f} seconds")
            logger.info(f"   Generation time: {generation_time:.2f} seconds")
            logger.info(f"   Real-time factor: {real_time_factor:.2f}x")
            logger.info(f"   Audio saved to: {output_file}")
            
            return True
        else:
            logger.error("❌ Model returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate speech with Dia model")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Path to quantized model directory")
    model_group.add_argument("--original", action="store_true", help="Use original model")
    
    # Text input
    parser.add_argument("--text", required=True, help="Text to convert to speech")
    parser.add_argument("--output", default="output.mp3", help="Output audio file")
    
    # Device and generation parameters
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.2, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, model_name = load_model(args.model, args.original, args.device)
        logger.info(f"✅ Loaded model: {model_name}")
        
        # Generate speech
        success = generate_speech(
            model, 
            args.text, 
            args.output,
            max_tokens=args.max_tokens,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
