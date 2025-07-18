#!/bin/bash

# Prepare repository for GitHub deployment
# This script initializes git, creates necessary files, and prepares for GitHub upload

set -e

echo "🚀 Preparing Dia-1.6B 4-Bit Quantization Repository for GitHub"
echo "============================================================="

# Check if we're already in a git repository
if [ -d ".git" ]; then
    echo "📁 Git repository already exists"
else
    echo "📁 Initializing Git repository..."
    git init
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# PyTorch
*.pth
*.pt
*.bin
*.safetensors

# Model files (too large for git)
quantized_models/
models/
checkpoints/
*.mp3
*.wav
*.flac
*.ogg

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
.tmp/

# Jupyter Notebooks
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Hugging Face cache
.cache/
transformers_cache/

# Large files that shouldn't be in git
*.zip
*.tar.gz
*.tar.bz2
*.7z

# Test outputs
test_output.*
output.*
benchmark_results.*
EOF
else
    echo "📝 .gitignore already exists"
fi

# Create LICENSE file
if [ ! -f "LICENSE" ]; then
    echo "📄 Creating LICENSE file..."
    cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Dia-1.6B 4-Bit Quantization Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

This project is based on the Dia-1.6B model by Nari Labs.
Original model: https://github.com/nari-labs/dia
Please refer to the original repository for the model's license terms.
EOF
else
    echo "📄 LICENSE already exists"
fi

# Create CONTRIBUTING.md
if [ ! -f "CONTRIBUTING.md" ]; then
    echo "📝 Creating CONTRIBUTING.md..."
    cat > CONTRIBUTING.md << 'EOF'
# Contributing to Dia-1.6B 4-Bit Quantization

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use the issue template if available
3. Provide detailed information:
   - Hardware specifications (GPU, VRAM, RAM)
   - Python and PyTorch versions
   - Error messages and stack traces
   - Steps to reproduce the issue

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/dia-tts-4bit.git
cd dia-tts-4bit

# Run setup script
./setup.sh

# Install development dependencies
pip install pytest flake8 black isort

# Run tests
python -m pytest tests/
```

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Testing

- Add tests for new features
- Ensure existing tests pass
- Test on different hardware configurations if possible
- Include performance benchmarks for quantization changes

### Documentation

- Update README.md for new features
- Add examples for new functionality
- Update configuration files
- Include hardware compatibility information

## Areas for Contribution

### High Priority
- [ ] Additional quantization methods (GGML, ONNX)
- [ ] Mobile deployment optimizations
- [ ] Better error handling and user feedback
- [ ] Performance optimizations

### Medium Priority
- [ ] Web interface (Gradio/Streamlit)
- [ ] Docker containers
- [ ] Cloud deployment guides
- [ ] More comprehensive benchmarks

### Low Priority
- [ ] Additional language support
- [ ] Custom voice training integration
- [ ] Advanced audio processing features

## Questions?

Feel free to open an issue for questions or join discussions in existing issues.

Thank you for contributing!
EOF
else
    echo "📝 CONTRIBUTING.md already exists"
fi

# Create issue templates directory
mkdir -p .github/ISSUE_TEMPLATE

# Bug report template
if [ ! -f ".github/ISSUE_TEMPLATE/bug_report.md" ]; then
    echo "🐛 Creating bug report template..."
    cat > .github/ISSUE_TEMPLATE/bug_report.md << 'EOF'
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With parameters '...'
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Error message**
```
Paste the full error message here
```

**Environment:**
- OS: [e.g. Ubuntu 20.04, Windows 11]
- Python version: [e.g. 3.10.6]
- PyTorch version: [e.g. 2.1.0]
- CUDA version: [e.g. 11.8]
- GPU: [e.g. RTX 3060 8GB]
- RAM: [e.g. 16GB]

**Additional context**
Add any other context about the problem here.
EOF
fi

# Feature request template
if [ ! -f ".github/ISSUE_TEMPLATE/feature_request.md" ]; then
    echo "✨ Creating feature request template..."
    cat > .github/ISSUE_TEMPLATE/feature_request.md << 'EOF'
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.

**Implementation ideas**
If you have ideas about how this could be implemented, please share them here.
EOF
fi

# Create pull request template
if [ ! -f ".github/pull_request_template.md" ]; then
    echo "🔄 Creating pull request template..."
    cat > .github/pull_request_template.md << 'EOF'
## Description
Brief description of the changes in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] I have tested these changes locally
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested on different hardware configurations (if applicable)

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Hardware Tested
- GPU: [e.g. RTX 3060 8GB]
- VRAM: [e.g. 8GB]
- RAM: [e.g. 16GB]
- OS: [e.g. Ubuntu 20.04]

## Performance Impact
Describe any performance changes (positive or negative) introduced by this PR.

## Additional Notes
Any additional information that reviewers should know.
EOF
fi

# Add all files to git
echo "📦 Adding files to git..."
git add .

# Check git status
echo "📊 Git status:"
git status

echo ""
echo "🎉 Repository prepared for GitHub!"
echo ""
echo "Next steps:"
echo "1. Review the files that will be committed:"
echo "   git status"
echo ""
echo "2. Commit the initial version:"
echo "   git commit -m 'Initial commit: Dia-1.6B 4-bit quantization project'"
echo ""
echo "3. Create a new repository on GitHub"
echo ""
echo "4. Add the remote origin:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/dia-tts-4bit.git"
echo ""
echo "5. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "6. Consider creating a release with pre-quantized models"
echo ""
echo "📋 Repository structure:"
find . -type f -name "*.py" -o -name "*.md" -o -name "*.json" -o -name "*.yml" -o -name "*.sh" | grep -v ".git" | sort
