# LiteAgent — Model Manifest

This document records the exact configuration, sizes, memory footprints, and download commands for the three model tiers used by LiteAgent.

## Target Model Tiers

| Tier | Model Name (Ollama Tag) | Quantization | Source / Reference URL | File Size | Expected VRAM/RAM Footprint | Verified (Yes/No + Method) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small (Edge)** | `llama3.2:1b` | Q4_0 / Q8_0 (standard Ollama 1B) | [Ollama Llama 3.2](https://ollama.com/library/llama3.2) | 1.3 GB | ~2.0 GB RAM (Pi 5 CPU or Workstation GPU) | Yes — `ollama list` size check (1.3 GB) |
| **Medium (Edge/Workstation)** | `llama3.2:3b` | Q4_0 / Q8_0 (standard Ollama 3B) | [Ollama Llama 3.2](https://ollama.com/library/llama3.2) | 2.0 GB | ~3.5 GB RAM/VRAM (Runs on Pi 5 CPU or Workstation GPU) | Yes — `ollama list` size check (2.0 GB) |
| **Large (Workstation)** | `llama3.1:8b` | Q4_0 (standard Ollama 8B) | [Ollama Llama 3.1](https://ollama.com/library/llama3.1) | 4.7 GB | ~8.0 GB VRAM (Runs on Workstation GPU) | **No** — `UNVERIFIED` (pending local pull and size check) |

## Model Download Instructions (For the user to run)

> [!IMPORTANT]
> Since model files are multi-gigabyte downloads, you should run these Ollama commands manually on your systems.

### 1. Workstation Setup (Local PC)
Run these commands on the workstation to pull all three tiers:
```powershell
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull llama3.1:8b
```

### 2. Edge Setup (Raspberry Pi 5)
Run these commands on the Raspberry Pi 5:
```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
```
*(Note: `llama3.1:8b` is excluded from the Pi 5 because it serves as the Workstation-only Large tier).*

## Verification Steps (After Pulling)

Once pulled, you can verify their sizes and quantizations by running:
```powershell
ollama show --modelfile llama3.2:1b
ollama show --modelfile llama3.2:3b
ollama show --modelfile llama3.1:8b
```
Please paste the output of these commands or confirm completion to mark them as `Verified` in this manifest.

---

## Dynamic GGUF Path Resolution for llama-cpp-python

Since `llama-cpp-python` loads raw GGUF files directly and cannot query the Ollama server API, LiteAgent implements a dynamic **Ollama Model Blob Resolver**. This utility allows the model store to remain unified under Ollama while enabling direct file loading.

### Mechanism
Ollama stores GGUFs as content-addressable blobs in its models directory (`D:\Ollama\Models\blobs\sha256-<hash>` on Windows, and `~/.ollama/models/blobs/sha256-<hash>` on the Pi). 
The resolver maps a model tag (e.g. `llama3.2:1b`) to its GGUF file path at runtime by:
1.  Locating the JSON manifest file under the manifests directory:
    `D:\Ollama\Models\manifests\registry.ollama.ai\library\<model_family>\<tag>`
2.  Reading the manifest JSON and extracting the `digest` hash of the layer with `"mediaType": "application/vnd.ollama.image.model"`.
3.  Constructing the GGUF file path:
    `D:\Ollama\Models\blobs\sha256-<digest_hash>`
4.  Passing this path directly to the `Llama` constructor of `llama-cpp-python`.

This dynamic mapping prevents disk space duplication and ensures that any models downloaded via Ollama can be executed instantly in our cache-critical path.
