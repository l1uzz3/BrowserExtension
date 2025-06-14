FROM python:3.10-slim

# Add essential build tools + Python headers
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    cmake \
    libboost-all-dev \
    opencl-headers \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY backend/ .

CMD ["bash"]
