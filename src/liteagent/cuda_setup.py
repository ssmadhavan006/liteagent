"""
Makes the pip-vendored CUDA 12 runtime visible to the llama.cpp shared library.

The prebuilt llama-cpp-python wheel is linked against CUDA 12.4 and loads
`cudart64_12.dll` / `cublas64_12.dll`. The workstation has the CUDA 13 toolkit
installed, whose DLLs carry different names, so those symbols are not otherwise
resolvable and `import llama_cpp` fails with a bare "could not find module".

Importing this module before `llama_cpp` registers the `nvidia/*/bin`
directories from the pip packages on the DLL search path. It is a no-op on
non-Windows platforms and whenever the packages are absent, so the edge device
(CPU-only, no CUDA wheels) is unaffected.
"""

import os
import sys

_configured = False


def configure_cuda_dll_path() -> list[str]:
    """Registers vendored CUDA DLL directories. Returns the paths added."""
    global _configured
    if _configured or not sys.platform.startswith("win"):
        return []

    added = []
    try:
        import nvidia
    except ImportError:
        _configured = True
        return []

    for pkg_root in nvidia.__path__:
        for component in ("cuda_runtime", "cublas"):
            bin_dir = os.path.join(pkg_root, component, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    continue
                # PATH matters too: the loader resolves transitive dependencies
                # of llama.dll through it, not through add_dll_directory alone.
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                added.append(bin_dir)

    _configured = True
    return added
