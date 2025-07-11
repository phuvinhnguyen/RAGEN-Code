#!/bin/bash

# Exit on error
set -e

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected"
        return 0
    else
        echo "No CUDA GPU detected"
        return 1
    fi
}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Main installation process
main() {
    # Install verl submodule in editable mode
    print_step "Setting up verl (git submodule)..."
    git submodule init
    git submodule update
    cd verl
    pip install -e . --no-dependencies # we put dependencies in requirements.txt
    cd ..
    
    # Install ragen in editable mode
    print_step "Installing ragen package..."
    pip install -e .
    
    # Install PyTorch and flash-attn with CUDA if available
    if check_cuda; then
        print_step "CUDA detected, checking CUDA version..."
        if command -v nvcc &> /dev/null; then
            nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            nvcc_major=$(echo $nvcc_version | cut -d. -f1)
            nvcc_minor=$(echo $nvcc_version | cut -d. -f2)
            print_step "Found NVCC version: $nvcc_version"
            if [[ "$nvcc_major" -gt 12 || ("$nvcc_major" -eq 12 && "$nvcc_minor" -ge 1) ]]; then
                print_step "CUDA $nvcc_version is already installed and meets requirements (>=12.4)"
                export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}
            else
                print_step "CUDA version < 12.4. Please install CUDA toolkit 12.4+ manually if needed."
                # Manual step: install CUDA toolkit 12.4+ if not present
            fi
        else
            print_step "NVCC not found. Please install CUDA toolkit 12.4+ manually if you want GPU support."
            # Manual step: install CUDA toolkit 12.4+ if not present
        fi
        print_step "Installing PyTorch with CUDA support..."
        pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
        print_step "Installing flash-attention..."
        pip3 install flash-attn==2.7.4.post1 --no-build-isolation
    else
        print_step "Installing PyTorch without CUDA support..."
        pip install torch==2.6.0
    fi
    
    # Install remaining requirements
    print_step "Installing additional requirements..."
    pip install -r requirements.txt

    # System dependencies (manual steps):
    echo -e "${BLUE}If you need faiss, openjdk, maven, or gdown, please install them manually:${NC}"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y default-jdk maven"
    echo "  pip install faiss-cpu gdown"
    echo "  # For CUDA toolkit, see: https://developer.nvidia.com/cuda-downloads"
    
    # Spacy models (manual step):
    echo -e "${BLUE}To use spacy, you may need to download models:${NC}"
    echo "  python -m spacy download en_core_web_sm"
    echo "  python -m spacy download en_core_web_lg"

    print_step "Downloading data..."
    python scripts/download_data.py

    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo "You are ready to use RAGEN in your current environment."
    echo "If you need system dependencies (faiss, openjdk, maven, gdown, CUDA), please install them manually as above."
}

# Run main installation
main
