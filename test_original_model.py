#!/usr/bin/env python3
"""
Test script for the original Dia-1.6B model.
This script will test the model performance and memory usage on the current hardware.
"""

import time
import psutil
import torch
import numpy as np
from dia.model import Dia

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory_usage():
    """Get GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def test_model_loading():
    """Test loading the original model and measure memory usage"""
    print("Testing Dia-1.6B model loading and inference...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Memory before loading
    mem_before = get_memory_usage()
    gpu_mem_before = get_gpu_memory_usage()
    print(f"Memory before loading: {mem_before:.1f} MB")
    print(f"GPU memory before loading: {gpu_mem_before:.1f} MB")
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    
    try:
        # Try different precisions to see what works on 4GB VRAM
        for dtype in ["float16", "bfloat16", "float32"]:
            print(f"\nTrying {dtype} precision...")
            try:
                model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype=dtype)
                load_time = time.time() - start_time
                
                # Memory after loading
                mem_after = get_memory_usage()
                gpu_mem_after = get_gpu_memory_usage()
                print(f"Model loaded successfully in {load_time:.2f} seconds")
                print(f"Memory after loading: {mem_after:.1f} MB (diff: +{mem_after - mem_before:.1f} MB)")
                print(f"GPU memory after loading: {gpu_mem_after:.1f} MB (diff: +{gpu_mem_after - gpu_mem_before:.1f} MB)")
                
                # Test inference
                test_text = "[S1] This is a test of the Dia text to speech model. [S2] How does it sound?"
                print(f"Testing inference with text: '{test_text}'")
                
                inference_start = time.time()
                output = model.generate(
                    test_text,
                    max_tokens=512,  # Short test
                    verbose=True,
                    use_torch_compile=False,
                    cfg_scale=3.0,
                    temperature=1.2,
                    top_p=0.95
                )
                inference_time = time.time() - inference_start
                
                print(f"Inference completed in {inference_time:.2f} seconds")
                if output is not None:
                    print(f"Generated audio length: {len(output)} samples")
                    print(f"Audio duration: {len(output) / 44100:.2f} seconds")
                    
                    # Save test output
                    model.save_audio("test_output.mp3", output)
                    print("Test audio saved as 'test_output.mp3'")
                
                # Memory after inference
                mem_final = get_memory_usage()
                gpu_mem_final = get_gpu_memory_usage()
                print(f"Memory after inference: {mem_final:.1f} MB")
                print(f"GPU memory after inference: {gpu_mem_final:.1f} MB")
                
                # Calculate performance metrics
                if output is not None:
                    audio_duration = len(output) / 44100
                    realtime_factor = audio_duration / inference_time
                    print(f"Real-time factor: {realtime_factor:.2f}x")
                    if realtime_factor < 1.0:
                        print("⚠️  Model is slower than real-time - quantization needed!")
                    else:
                        print("✅ Model runs faster than real-time")
                
                return True, dtype, mem_after - mem_before, gpu_mem_after - gpu_mem_before, inference_time
                
            except Exception as e:
                print(f"Failed with {dtype}: {e}")
                continue
                
    except Exception as e:
        print(f"Error loading model: {e}")
        return False, None, 0, 0, 0
    
    print("❌ Failed to load model with any precision")
    return False, None, 0, 0, 0

if __name__ == "__main__":
    success, dtype, ram_usage, vram_usage, inference_time = test_model_loading()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if success:
        print(f"✅ Model loaded successfully with {dtype} precision")
        print(f"📊 RAM usage: {ram_usage:.1f} MB")
        print(f"🎮 VRAM usage: {vram_usage:.1f} MB")
        print(f"⏱️  Inference time: {inference_time:.2f} seconds")
        
        if vram_usage > 4000:  # 4GB = 4000MB
            print("⚠️  Model uses more than 4GB VRAM - quantization strongly recommended")
        elif vram_usage > 3000:  # 3GB
            print("⚠️  Model uses significant VRAM - quantization recommended")
        else:
            print("✅ Model fits comfortably in 4GB VRAM")
            
    else:
        print("❌ Failed to load model")
        print("This indicates the model requires more than available VRAM")
        print("Quantization is necessary to run on this hardware")
