#!/usr/bin/env python3
"""
Benchmark script for comparing original vs quantized Dia models

This script compares performance metrics between the original and quantized models:
- Memory usage (RAM and VRAM)
- Inference speed
- Audio quality (basic metrics)
- Real-time factor
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import numpy as np
import psutil

# Add dia to path
sys.path.append('./dia')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark class for model performance testing"""
    
    def __init__(self):
        self.results = {}
        self.test_texts = [
            "[S1] This is a short test sentence.",
            "[S1] This is a medium length test sentence with more words to evaluate performance. [S2] How does it sound?",
            "[S1] This is a longer test sentence that contains multiple clauses and should provide a good evaluation of the model's performance across different sentence lengths and complexity levels. [S2] The audio quality should remain consistent throughout this longer generation.",
        ]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        ram_mb = process.memory_info().rss / 1024 / 1024
        
        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return {"ram_mb": ram_mb, "vram_mb": vram_mb}
    
    def benchmark_model(self, model, model_name: str, device: str = "cpu") -> Dict[str, Any]:
        """Benchmark a single model"""
        logger.info(f"Benchmarking {model_name}...")
        
        results = {
            "model_name": model_name,
            "device": device,
            "torch_version": torch.__version__,
            "tests": []
        }
        
        # Memory before inference
        memory_before = self.get_memory_usage()
        
        for i, text in enumerate(self.test_texts):
            logger.info(f"Testing text {i+1}/{len(self.test_texts)}: {text[:50]}...")
            
            test_result = {
                "text_length": len(text),
                "text": text,
                "success": False,
                "inference_time": 0,
                "audio_length": 0,
                "audio_duration": 0,
                "real_time_factor": 0,
                "memory_peak": {"ram_mb": 0, "vram_mb": 0},
                "error": None
            }
            
            try:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Measure inference time
                start_time = time.time()
                
                # Generate audio
                output = model.generate(
                    text,
                    max_tokens=min(512, len(text) * 4),  # Adaptive max tokens
                    verbose=False,
                    use_torch_compile=False,
                    cfg_scale=3.0,
                    temperature=1.2,
                    top_p=0.95
                )
                
                inference_time = time.time() - start_time
                
                # Memory after inference
                memory_after = self.get_memory_usage()
                
                if output is not None:
                    audio_length = len(output)
                    audio_duration = audio_length / 44100  # Assuming 44.1kHz sample rate
                    real_time_factor = audio_duration / inference_time if inference_time > 0 else 0
                    
                    test_result.update({
                        "success": True,
                        "inference_time": inference_time,
                        "audio_length": audio_length,
                        "audio_duration": audio_duration,
                        "real_time_factor": real_time_factor,
                        "memory_peak": {
                            "ram_mb": max(memory_before["ram_mb"], memory_after["ram_mb"]),
                            "vram_mb": max(memory_before["vram_mb"], memory_after["vram_mb"])
                        }
                    })
                    
                    logger.info(f"✅ Success: {inference_time:.2f}s, RTF: {real_time_factor:.2f}x")
                else:
                    test_result["error"] = "Model returned None"
                    logger.warning("⚠️  Model returned None")
                
            except Exception as e:
                test_result["error"] = str(e)
                logger.error(f"❌ Error: {e}")
            
            results["tests"].append(test_result)
        
        # Calculate summary statistics
        successful_tests = [t for t in results["tests"] if t["success"]]
        
        if successful_tests:
            results["summary"] = {
                "success_rate": len(successful_tests) / len(results["tests"]),
                "avg_inference_time": np.mean([t["inference_time"] for t in successful_tests]),
                "avg_real_time_factor": np.mean([t["real_time_factor"] for t in successful_tests]),
                "max_ram_mb": max([t["memory_peak"]["ram_mb"] for t in successful_tests]),
                "max_vram_mb": max([t["memory_peak"]["vram_mb"] for t in successful_tests]),
                "total_audio_duration": sum([t["audio_duration"] for t in successful_tests]),
                "total_inference_time": sum([t["inference_time"] for t in successful_tests])
            }
        else:
            results["summary"] = {
                "success_rate": 0,
                "error": "All tests failed"
            }
        
        return results
    
    def load_and_benchmark_original(self, device: str = "cpu") -> Optional[Dict[str, Any]]:
        """Load and benchmark the original model"""
        try:
            from dia.model import Dia
            
            logger.info("Loading original Dia-1.6B model...")
            model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
            
            return self.benchmark_model(model, "Original Dia-1.6B", device)
            
        except Exception as e:
            logger.error(f"Failed to load original model: {e}")
            return None
    
    def load_and_benchmark_quantized(self, model_path: str, device: str = "cpu") -> Optional[Dict[str, Any]]:
        """Load and benchmark a quantized model"""
        try:
            model_path = Path(model_path)
            
            # Try different loading methods
            model = None
            
            # Method 1: Load full model
            if (model_path / "model.pth").exists():
                logger.info(f"Loading quantized model from {model_path}/model.pth...")
                model = torch.load(model_path / "model.pth", map_location=device)
            
            # Method 2: Load state dict
            elif (model_path / "pytorch_model.bin").exists():
                logger.info(f"Loading quantized model state dict from {model_path}...")
                from dia.model import Dia
                
                model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
                state_dict = torch.load(model_path / "pytorch_model.bin", map_location=device)
                model.load_state_dict(state_dict)
            
            else:
                raise FileNotFoundError(f"No model files found in {model_path}")
            
            model_name = f"Quantized ({model_path.name})"
            return self.benchmark_model(model, model_name, device)
            
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            return None
    
    def compare_models(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model results"""
        if len(results) < 2:
            return {"error": "Need at least 2 models to compare"}
        
        comparison = {
            "models": [r["model_name"] for r in results],
            "comparison": {}
        }
        
        # Find baseline (usually the original model)
        baseline = results[0]
        for result in results:
            if "Original" in result["model_name"]:
                baseline = result
                break
        
        baseline_summary = baseline.get("summary", {})
        
        for result in results:
            model_name = result["model_name"]
            summary = result.get("summary", {})
            
            if not summary or "success_rate" not in summary:
                comparison["comparison"][model_name] = {"error": "No valid summary"}
                continue
            
            comp = {
                "success_rate": summary["success_rate"],
                "memory_usage": {
                    "ram_mb": summary.get("max_ram_mb", 0),
                    "vram_mb": summary.get("max_vram_mb", 0)
                }
            }
            
            # Calculate relative performance
            if baseline_summary and baseline_summary.get("success_rate", 0) > 0:
                comp["relative_performance"] = {
                    "inference_speed": (baseline_summary.get("avg_inference_time", 1) / 
                                      summary.get("avg_inference_time", 1)),
                    "memory_efficiency": {
                        "ram": (baseline_summary.get("max_ram_mb", 1) / 
                               summary.get("max_ram_mb", 1)),
                        "vram": (baseline_summary.get("max_vram_mb", 1) / 
                                summary.get("max_vram_mb", 1))
                    }
                }
            
            comparison["comparison"][model_name] = comp
        
        return comparison
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save benchmark results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        benchmark_data = {
            "timestamp": time.time(),
            "system_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
            },
            "results": results,
            "comparison": self.compare_models(results) if len(results) > 1 else None
        }
        
        with open(output_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Dia models")
    parser.add_argument("--original", action="store_true", help="Benchmark original model")
    parser.add_argument("--quantized", nargs="+", help="Paths to quantized models")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    
    args = parser.parse_args()
    
    if not args.original and not args.quantized:
        logger.error("Must specify --original and/or --quantized models")
        return 1
    
    benchmark = ModelBenchmark()
    results = []
    
    # Benchmark original model
    if args.original:
        original_result = benchmark.load_and_benchmark_original(args.device)
        if original_result:
            results.append(original_result)
    
    # Benchmark quantized models
    if args.quantized:
        for model_path in args.quantized:
            quantized_result = benchmark.load_and_benchmark_quantized(model_path, args.device)
            if quantized_result:
                results.append(quantized_result)
    
    if not results:
        logger.error("No models were successfully benchmarked")
        return 1
    
    # Save results
    benchmark.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for result in results:
        summary = result.get("summary", {})
        print(f"\n📊 {result['model_name']}")
        print(f"   Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"   Avg Inference Time: {summary.get('avg_inference_time', 0):.2f}s")
        print(f"   Avg Real-time Factor: {summary.get('avg_real_time_factor', 0):.2f}x")
        print(f"   Max RAM Usage: {summary.get('max_ram_mb', 0):.1f} MB")
        print(f"   Max VRAM Usage: {summary.get('max_vram_mb', 0):.1f} MB")
    
    print(f"\n📄 Detailed results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
