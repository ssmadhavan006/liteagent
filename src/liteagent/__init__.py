"""LiteAgent: complexity-aware routing and tiered KV-cache co-design."""

# Must run before any `import llama_cpp` anywhere in the process: the CUDA build
# cannot resolve its DLLs otherwise. No-op off Windows and without the vendored
# CUDA packages, so the CPU-only edge device is unaffected.
from liteagent.cuda_setup import configure_cuda_dll_path

configure_cuda_dll_path()
