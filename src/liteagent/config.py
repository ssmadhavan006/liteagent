import os

# System & Package Constants
LITEAGENT_VERSION = "0.1.0"
PROTOCOL_VERSION = 1

# GPU offload, per device role.
#
# Both default to 0 (CPU). The edge target (Raspberry Pi 5) has no usable GPU.
# The workstation does have an RTX 5070, but the available prebuilt CUDA build
# (llama-cpp-python 0.3.4 / cu124) aborts in ggml-cuda partway through decoding
# on Blackwell, so GPU offload is opt-in rather than default until a llama.cpp
# with sm_120 support is available. See architecture.md 10.2.
#
# Consequence: nvidia-smi energy sampling records idle draw while offload is 0,
# so energy_joules is not a valid inference-energy measurement in this mode.
EDGE_N_GPU_LAYERS = int(os.environ.get("LITEAGENT_EDGE_GPU_LAYERS", "0"))
WORKSTATION_N_GPU_LAYERS = int(os.environ.get("LITEAGENT_WORKSTATION_GPU_LAYERS", "0"))

# Default Model & Context Configurations
DEFAULT_CTX_SIZE = 4096
DEFAULT_MODEL_TAG = "llama3.1:8b"
DEFAULT_EDGE_MEDIUM_TAG = "llama3.2:3b"
DEFAULT_EDGE_SMALL_TAG = "llama3.2:1b"

# Default Network Configurations
DEFAULT_WORKSTATION_HOST = "127.0.0.1"
DEFAULT_WORKSTATION_PORT = 50051
DEFAULT_CONNECT_TIMEOUT = 180.0
MAX_TOKENS_CAP = 2048
MAX_PROMPT_LEN = 32768
