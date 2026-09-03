FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Remove pre-installed torchvision/torchaudio — not needed for text-only LLM inference
RUN pip uninstall -y torchvision torchaudio 2>/dev/null || true

# Install vLLM + Python deps (torch 2.8.0 + CUDA 12.8.1 in base image)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ===============================
# DOWNLOAD granite-4.1-8b-instruct
# ===============================
RUN python3 -u <<'EOF'
from huggingface_hub import snapshot_download

print("Downloading ibm-granite/granite-4.1-8b-instruct...", flush=True)

snapshot_download(
    repo_id="ibm-granite/granite-4.1-8b-instruct",
    local_dir="/app/models/granite-4.1-8b-instruct",
    local_dir_use_symlinks=False,
    resume_download=True
)

print("granite-4.1-8b-instruct download complete", flush=True)
EOF

WORKDIR /app
COPY handler.py /app/handler.py

ENTRYPOINT ["python3"]
CMD ["-u", "handler.py"]