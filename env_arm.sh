#!/bin/bash
set -e

echo "==== Step 0===="
sudo apt-get install -y \
    libsuitesparse-dev \
    libmetis-dev \
    libgomp1 \
    build-essential \
    python3-dev \
    gdal-bin \
    libgdal-dev \
    libblas3 \
    libatlas-base-dev \
    libaio-dev \
    util-linux

echo "==== Step 1: Install Fiona ===="
conda install -y fiona

echo "==== Step 2: Upgrade pip ===="
pip install pip==24.0 --no-cache-dir

echo "==== Step 3: Link TensorRT and OpenCV from system to Conda env ===="
# Replace with your actual Python version if not 3.8
PYVER=3.8
CONDA_SITEPKG="$CONDA_PREFIX/lib/python${PYVER}/site-packages"

# Tensorrt
if [ -e "${CONDA_SITEPKG}/tensorrt" ]; then
    echo "tensorrt already linked, skipping."
else
    echo "Linking tensorrt..."
    ln -sf "/usr/lib/python${PYVER}/dist-packages/tensorrt" "${CONDA_SITEPKG}/tensorrt"
fi

# OpenCV (cv2)
if [ -e "${CONDA_SITEPKG}/cv2" ]; then
    echo "cv2 already linked, skipping."
else
    echo "Linking cv2..."
    ln -sf "/usr/lib/python${PYVER}/dist-packages/cv2" "${CONDA_SITEPKG}/cv2"
fi

echo "==== Step 4: Install Jetson-compatible Deep Learning packages ===="
pip install \
  --index-url https://pypi.jetson-ai-lab.dev/jp5/cu114/+simple \
  bitsandbytes==0.39.1 \
  onnxruntime_gpu==1.16.3 \
  pycuda==2024.1 \
  torch==2.2.0 \
  torchaudio==2.2.2+cefdb36 \
  torchvision==0.17.2+c1d70fe \
  --no-cache-dir

echo "==== Step 5: Install Model Zoo and Optimization Toolkits ===="
pip install \
    timm==1.0.15 \
    --upgrade-strategy only-if-needed --no-cache-dir

pip install --upgrade py-cpuinfo --no-cache-dir

pip install \
    theseus-ai \
    transformers \
    pytorch_lightning \
    accelerate \
    peft==0.6.0 \
    tensorboard \
    --no-cache-dir

echo "==== Step 6 ===="
pip install \
    scikit-learn \
    aioboto3 \
    aiofiles \
    bokeh==2.4.3 \
    boto3==1.24.59 \
    cachetools \
    casadi \
    control==0.9.1 \
    coverage \
    docker \
    geopandas>=0.12.1 \
    grpcio==1.43.0 \
    grpcio-tools==1.43.0 \
    guppy3==3.1.2 \
    hydra-core==1.1.0rc1 \
    hypothesis \
    joblib \
    jupyter \
    jupyterlab \
    matplotlib \
    mock \
    moto \
    nest_asyncio \
    numpy==1.23.4 \
    pandas \
    Pillow \
    pre-commit \
    psutil \
    pyarrow \
    pyinstrument \
    pyogrio \
    pyquaternion>=0.9.5 \
    pytest \
    rasterio \
    ray \
    requests \
    retry \
    rtree \
    s3fs \
    scipy \
    selenium \
    setuptools==59.5.0 \
    Shapely>=2.0.0 \
    SQLAlchemy==1.4.27 \
    sympy \
    testbook \
    tornado \
    tqdm \
    typer \
    ujson \
    urllib3 \
    pydantic \
    sentencepiece \
    einops \
    future \
    onnx_graphsurgeon \
    polygraphy
    --no-cache-dir

echo "All steps completed successfully!"
