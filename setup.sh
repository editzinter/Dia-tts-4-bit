#!/bin/bash

# Dia-1.6B 4-Bit Quantization Setup Script
# This script sets up the environment for quantizing the Dia-1.6B model

set -e  # Exit on any error

echo "🚀 Setting up Dia-1.6B 4-Bit Quantization Environment"
echo "=================================================="

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "⚡ Found uv - using fast installation method"
    USE_UV=true
else
    echo "📦 uv not found - using traditional pip method"
    echo "💡 For faster installation, consider installing uv: https://github.com/astral-sh/uv"
    USE_UV=false
fi

# Check if Python is available (only needed for pip method)
if [ "$USE_UV" = false ]; then
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
        exit 1
    fi

    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "🐍 Python version: $python_version"

    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        echo "❌ Python 3.8 or later is required. Current version: $python_version"
        exit 1
    fi
fi

# Setup environment based on available tools
if [ "$USE_UV" = true ]; then
    echo "⚡ Setting up with uv..."

    # Create pyproject.toml if it doesn't exist
    if [ ! -f "pyproject.toml" ]; then
        echo "📝 Creating pyproject.toml for uv..."
        cat > pyproject.toml << 'EOF'
[project]
name = "dia-tts-4bit"
version = "1.0.0"
description = "4-bit quantized Dia-1.6B TTS model for efficient inference"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "transformers>=4.30.0",
    "huggingface-hub>=0.15.0",
    "numpy>=1.21.0",
    "soundfile>=0.12.0",
    "psutil>=5.8.0",
    "tqdm>=4.64.0",
]

[project.optional-dependencies]
advanced = [
    "bitsandbytes>=0.41.0",
    "auto-gptq>=0.4.0",
    "optimum>=1.12.0",
    "accelerate>=0.20.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
EOF
    fi

    # Install with uv
    echo "📦 Installing dependencies with uv..."
    uv sync

    # Activate uv environment
    echo "🔧 Activating uv environment..."
    source .venv/bin/activate

else
    echo "📦 Setting up with pip..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    else
        echo "📦 Virtual environment already exists"
    fi

    # Activate virtual environment
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip

    # Install PyTorch (CPU version for compatibility)
    echo "🔥 Installing PyTorch..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Install basic requirements
    echo "📋 Installing requirements..."
    pip install -r requirements.txt
fi

# Clone and install Dia model if not already present
if [ ! -d "dia" ]; then
    echo "📥 Cloning Dia repository..."
    git clone https://github.com/nari-labs/dia.git
else
    echo "📁 Dia repository already exists"
fi

echo "🔧 Installing Dia model..."
cd dia
pip install -e .
cd ..

# Create directories for quantized models
echo "📁 Creating directories..."
mkdir -p quantized_models
mkdir -p examples
mkdir -p benchmarks

# Create a simple example script
cat > examples/basic_quantization.py << 'EOF'
#!/usr/bin/env python3
"""
Basic quantization example for Dia-1.6B model
"""

import sys
import os
sys.path.append('..')

from simple_quantize import SimpleDiaQuantizer

def main():
    print("🚀 Starting basic quantization example...")
    
    # Initialize quantizer
    quantizer = SimpleDiaQuantizer("nari-labs/Dia-1.6B")
    
    # Load model (use CPU to avoid VRAM issues)
    print("📥 Loading model...")
    if not quantizer.load_model(device="cpu"):
        print("❌ Failed to load model")
        return
    
    # Apply 4-bit quantization
    print("🔧 Applying 4-bit quantization...")
    if not quantizer.quantize_linear_layers():
        print("❌ Quantization failed")
        return
    
    # Save quantized model
    print("💾 Saving quantized model...")
    output_dir = "../quantized_models/dia-1.6b-4bit-example"
    if not quantizer.save_quantized_model(output_dir, "4bit"):
        print("❌ Failed to save model")
        return
    
    print("✅ Quantization completed successfully!")
    print(f"📁 Quantized model saved to: {output_dir}")

if __name__ == "__main__":
    main()
EOF

# Make example executable
chmod +x examples/basic_quantization.py

# Create a test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify installation
"""

import sys
import torch

def test_installation():
    print("🧪 Testing installation...")
    
    # Test PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    # Test Dia import
    try:
        sys.path.append('./dia')
        from dia.model import Dia
        print("✅ Dia model import successful")
    except ImportError as e:
        print(f"❌ Dia model import failed: {e}")
        return False
    
    # Test quantization scripts
    try:
        from simple_quantize import SimpleDiaQuantizer
        print("✅ Simple quantization script import successful")
    except ImportError as e:
        print(f"❌ Simple quantization import failed: {e}")
        return False
    
    print("🎉 All tests passed! Installation is ready.")
    return True

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
EOF

# Make test script executable
chmod +x test_installation.py

# Run installation test
echo "🧪 Testing installation..."
python test_installation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    if [ "$USE_UV" = true ]; then
        echo "1. Activate the environment: source .venv/bin/activate"
        echo "2. For GPU support: uv add torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else
        echo "1. Activate the virtual environment: source venv/bin/activate"
        echo "2. For GPU support: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
    fi
    echo "3. Run basic quantization: python examples/basic_quantization.py"
    echo "4. Or use presets: python preset_quantize.py --preset ultra_low_vram --output ./my_quantized_model"
    echo "5. Generate speech: python examples/generate_speech.py --model ./my_quantized_model --text 'Hello world'"
    echo ""
    echo "📚 See README.md for detailed usage instructions"
    echo ""
else
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi
