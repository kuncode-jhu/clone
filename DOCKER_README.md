# Brain-to-Text Docker Setup

This Docker setup provides a complete environment for the brain-to-text neuroprosthesis project with automatic data downloading and dual Python environments.

## Quick Start

### Prerequisites
- Docker with GPU support (NVIDIA Container Toolkit)
- NVIDIA GPU with CUDA support
- At least 100GB free disk space (for datasets)

### Build and Run
```bash
# Build the Docker image
docker-compose build

# Start the container (interactive mode)
docker-compose up -d
docker-compose exec brain2text bash

# OR run directly with Docker
docker build -t brain2text .
docker run --gpus all -it --rm -v $(pwd)/results:/workspace/results brain2text
```

## What's Included

### Dual Python Environments
- **Main Environment** (`b2txt25`): Python 3.10, PyTorch with CUDA 12.1
- **Language Model Environment** (`b2txt25_lm`): Python 3.9, PyTorch 1.13.1

### Automatic Data Download
- Downloads from Dryad: Neural data, pretrained models (~11GB)
- Downloads from Google Drive: Language models (~52GB) 
- Auto-extracts archives as needed

### System Components
- CUDA 12.1 development environment
- Redis server (for inter-process communication)
- All required build tools (CMake, GCC, etc.)

## Usage Examples

### Interactive Development
```bash
# Start container and enter main environment
docker-compose up -d
docker-compose exec brain2text bash

# Switch environments inside container
conda activate b2txt25        # Main training environment
conda activate b2txt25_lm     # Language model environment
```

### Run Specific Scripts
```bash
# Train a model (main environment)
docker run --gpus all -it brain2text ./run-main.sh python train_model.py

# Run language model inference (LM environment) 
docker run --gpus all -it brain2text ./run-lm.sh python language-model-standalone.py

# Run data download only
docker run --gpus all -it brain2text python download_data_enhanced.py
```

### Jupyter Notebook
```bash
# Start container with Jupyter
docker-compose up -d
docker-compose exec brain2text bash -c "conda activate b2txt25 && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"

# Access at: http://localhost:8888
```

## Directory Structure

```
/workspace/                    # Container working directory
├── data/                     # Datasets (auto-downloaded)
├── model_training/           # Training scripts
├── language_model/           # Language model code  
├── analyses/                 # Analysis notebooks
├── results/                  # Output directory (mounted)
├── start.sh                  # Container startup script
├── run-main.sh              # Main environment runner
└── run-lm.sh                # LM environment runner
```

## Troubleshooting

### GPU Issues
```bash
# Check GPU access inside container
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Data Download Issues
```bash
# Manual data download
docker-compose exec brain2text python download_data_enhanced.py

# Check disk space
docker-compose exec brain2text df -h
```

### Environment Issues
```bash
# List conda environments
docker-compose exec brain2text conda env list

# Test environments
docker-compose exec brain2text ./run-main.sh python -c "import torch; print(torch.version.cuda)"
docker-compose exec brain2text ./run-lm.sh python -c "import torch; print(torch.__version__)"
```

## For Lambda Labs

### Building on Lambda Labs
```bash
# Clone repo
git clone [your-repo-url]
cd nejm-brain-to-text

# Build (will take 20-30 minutes)
docker build -t brain2text .

# Run with GPU support
docker run --gpus all -it --shm-size=8g brain2text
```

### Persistent Storage on Lambda Labs
```bash
# Mount external storage for data persistence
docker run --gpus all -it --shm-size=8g \
  -v /path/to/storage:/workspace/data \
  brain2text
```

## Performance Notes

- **Build time**: ~20-30 minutes (mostly Python package installations)
- **Image size**: ~8-10GB (without data)
- **Runtime memory**: 16GB+ recommended (language models are large)
- **Storage**: 100GB+ required for all datasets

## Ready to Go!

Once built, this container provides a complete, reproducible environment for brain-to-text research with automatic data management and dual Python environments.
