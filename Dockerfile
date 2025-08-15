# Brain-to-Text Neuroprosthesis Docker Environment
# Multi-stage build to handle dual Python environments and large datasets

FROM ubuntu:22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    gcc \
    g++ \
    # System utilities
    wget \
    curl \
    git \
    vim \
    htop \
    # rclone CLI for Drive/Cloud syncing
    rclone \
    # Redis for inter-process communication
    redis-server \
    # Python build dependencies
    python3-dev \
    python3-pip \
    # Additional dependencies
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Initialize conda
RUN conda init bash

WORKDIR /workspace

# Copy source code (but not data directory)
COPY . /workspace/
RUN rm -rf /workspace/data/*

# Create data directory with proper structure
RUN mkdir -p /workspace/data

# ================================================================================
# Stage 1: Main training environment (b2txt25)
# ================================================================================

FROM base AS main_env

# Create main conda environment using conda-forge channel
RUN conda create -n b2txt25 -c conda-forge python=3.10 -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "b2txt25", "/bin/bash", "-c"]

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install main environment dependencies
RUN pip install \
    redis==5.2.1 \
    jupyter==1.1.1 \
    numpy==2.1.2 \
    pandas==2.3.0 \
    pyarrow==16.1.0 \
    matplotlib==3.10.1 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    tqdm==4.67.1 \
    g2p_en==2.1.0 \
    h5py==3.13.0 \
    omegaconf==2.3.0 \
    editdistance==0.8.1 \
    huggingface-hub==0.33.1 \
    transformers==4.53.0 \
    tokenizers==0.21.2 \
    accelerate==1.8.1 \
    bitsandbytes==0.46.0 \
    gdown

# Install the local package
RUN pip install -e .

# ================================================================================
# Stage 2: Language model environment (b2txt25_lm) 
# ================================================================================

FROM main_env AS lm_env

# Create language model environment using conda-forge
RUN conda create -n b2txt25_lm -c conda-forge python=3.9 -y

# Switch to language model environment
SHELL ["conda", "run", "-n", "b2txt25_lm", "/bin/bash", "-c"]

# Upgrade pip
RUN pip install --upgrade pip

# Install language model environment dependencies (older torch version)
RUN pip install \
    torch==1.13.1 \
    redis==5.0.6 \
    jupyter==1.1.1 \
    numpy==1.24.4 \
    matplotlib==3.9.0 \
    scipy==1.11.1 \
    scikit-learn==1.6.1 \
    tqdm==4.66.4 \
    g2p_en==2.1.0 \
    omegaconf==2.3.0 \
    huggingface-hub==0.23.4 \
    transformers==4.40.0 \
    tokenizers==0.19.1 \
    accelerate==0.33.0 \
    bitsandbytes==0.41.1 \
    gdown

# Build language model components (Kaldi-based)
RUN cd language_model/runtime/server/x86 && python setup.py install

# ================================================================================
# Final stage: Runtime environment
# ================================================================================

FROM lm_env AS runtime

# Create startup script
RUN echo '#!/bin/bash\n\
# Start Redis server\n\
redis-server --daemonize yes\n\
\n\
# Function to activate environment and run command\n\
activate_main() {\n\
    source /opt/conda/etc/profile.d/conda.sh\n\
    conda activate b2txt25\n\
}\n\
\n\
activate_lm() {\n\
    source /opt/conda/etc/profile.d/conda.sh\n\
    conda activate b2txt25_lm\n\
}\n\
\n\
# Download data if not present\n\
if [ ! -f "/workspace/data/languageModel.tar.gz" ] || [ ! -f "/workspace/data/t15_copyTask.pkl" ]; then\n\
    echo "Downloading datasets..."\n\
    activate_main\n\
    python download_data_enhanced.py\n\
    echo "Data download completed!"\n\
fi\n\
\n\
# Extract language models if needed\n\
cd /workspace/data\n\
if [ -f "languageModel.tar.gz" ] && [ ! -d "openwebtext_3gram_lm_sil" ]; then\n\
    echo "Extracting 3-gram language model..."\n\
    tar -xzf languageModel.tar.gz\n\
fi\n\
\n\
if [ -f "languageModel_5gram.tar.gz" ] && [ ! -d "openwebtext_5gram_lm_sil" ]; then\n\
    echo "Extracting 5-gram language model..."\n\
    tar -xzf languageModel_5gram.tar.gz\n\
fi\n\
\n\
cd /workspace\n\
\n\
# If no arguments provided, start interactive bash with main environment\n\
if [ $# -eq 0 ]; then\n\
    echo "Starting brain-to-text environment..."\n\
    echo "Available environments:"\n\
    echo "   - Main training: conda activate b2txt25"\n\
    echo "   - Language model: conda activate b2txt25_lm"\n\
    echo ""\n\
    echo "Data directory: /workspace/data"\n\
    echo "Ready for brain-to-text experiments!"\n\
    source /opt/conda/etc/profile.d/conda.sh\n\
    conda activate b2txt25\n\
    exec /bin/bash\n\
else\n\
    # Execute provided command\n\
    exec "$@"\n\
fi' > /workspace/start.sh && chmod +x /workspace/start.sh

# Create convenience scripts for each environment
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate b2txt25\n\
exec "$@"' > /workspace/run-main.sh && chmod +x /workspace/run-main.sh

RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate b2txt25_lm\n\
exec "$@"' > /workspace/run-lm.sh && chmod +x /workspace/run-lm.sh

# Set working directory
WORKDIR /workspace

# Expose Jupyter port (optional)
EXPOSE 8888

# Set entrypoint
ENTRYPOINT ["/workspace/start.sh"]
CMD []
