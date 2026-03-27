# Base image with GPU support
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Set CUDA and NVIDIA library paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/lib:${LD_LIBRARY_PATH}

RUN ldconfig

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Set environment variables for headless rendering with EGL
ENV PYOPENGL_PLATFORM=egl
ENV DISPLAY=:99

RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox

# Install software-properties-common first to enable adding PPAs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa -y

# Install system-level dependencies, including Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    bison \
    flex \
    libncurses5-dev \
    libncursesw5-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    git \
    wget \
    unzip \
    ca-certificates \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libgl1 \
    libglu1-mesa \
    mesa-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py



# --- Robust runtime fix for libcuda.so (covers /usr/lib64 and RO mounts) ---
RUN cat > /usr/local/bin/fix-cuda.sh <<'BASH' && chmod +x /usr/local/bin/fix-cuda.sh
#!/usr/bin/env bash
set -euo pipefail

candidates=(
  /usr/local/nvidia/lib64
  /usr/lib/x86_64-linux-gnu
  /lib/x86_64-linux-gnu
  /usr/lib64
  /usr/lib/wsl/lib
)

found_lib=""
for d in "${candidates[@]}"; do
  if [ -e "$d/libcuda.so.1" ]; then
    found_lib="$d/libcuda.so.1"
    break
  fi
done

# As a last resort, ask the loader cache
if [ -z "$found_lib" ]; then
  if path=$(ldconfig -p | awk '/libcuda\.so\.1/{print $NF; exit}'); then
    found_lib="$path"
  fi
fi

if [ -n "$found_lib" ]; then
  target_dir="$(dirname "$found_lib")"
  # If we can write next to libcuda.so.1, create the symlink there; else use /usr/local/lib
  if [ -w "$target_dir" ]; then
    [ -e "$target_dir/libcuda.so" ] || ln -s "$found_lib" "$target_dir/libcuda.so"
    echo "$target_dir" > /etc/ld.so.conf.d/nvidia.conf
  else
    mkdir -p /usr/local/lib
    [ -e /usr/local/lib/libcuda.so ] || ln -s "$found_lib" /usr/local/lib/libcuda.so
    echo "/usr/local/lib" > /etc/ld.so.conf.d/nvidia.conf
  fi
  ldconfig || true
fi

# diagnostics (non-fatal)
ldconfig -p | grep -E 'libcuda\.so(\.1)?' || true
for d in "${candidates[@]}" /usr/local/lib; do
  [ -d "$d" ] && ls -l "$d"/libcuda.so* 2>/dev/null || true
done

exec "$@"
BASH
# --- End of libcuda.so fix ---



# Set the working directory inside the container
WORKDIR /hyperagents

# Copy the entire repository into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install proofgrader for imo_proof domain
RUN pip install -e proofgrader_repo

# Download things for balrog domains
RUN python -m domains.balrog.scripts.post_install

# For Genesis: install PyTorch with CUDA 13.0 support
# First check the Cuda version: nvidia-smi
# For other CUDA versions, adjust the --index-url accordingly (e.g., cu118, cu121, cu124)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Ensure the libcuda symlink fix runs before your command
ENTRYPOINT ["/usr/local/bin/fix-cuda.sh"]

# Keep the container running by default
CMD ["tail", "-f", "/dev/null"]
