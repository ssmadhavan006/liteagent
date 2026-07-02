# LiteAgent — Model Manifest

This document records the exact configuration, sizes, memory footprints, and download commands for the three model tiers used by LiteAgent.

## Target Model Tiers

| Tier | Model Name (Ollama Tag) | Quantization | Source / Reference URL | File Size | Expected VRAM/RAM Footprint | Verified (Yes/No + Method) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small (Edge)** | `llama3.2:1b` | Q4_0 / Q8_0 (standard Ollama 1B) | [Ollama Llama 3.2](https://ollama.com/library/llama3.2) | 1.3 GB | ~2.0 GB RAM (Pi 5 CPU or Workstation GPU) | **No** — `UNVERIFIED` (pending local pull and size check) |
| **Medium (Edge/Workstation)** | `llama3.1:8b` | Q4_0 (standard Ollama 8B) | [Ollama Llama 3.1](https://ollama.com/library/llama3.1) | 4.7 GB | ~8.0 GB RAM/VRAM (Runs on Pi 5 CPU or Workstation GPU) | **No** — `UNVERIFIED` (pending local pull and size check) |
| **Large (Workstation)** | `qwen2.5:14b` | Q4_0 / Q4_K_M | [Ollama Qwen 2.5](https://ollama.com/library/qwen2.5) | 9.0 GB | ~12.0 GB VRAM (Runs on Workstation GPU) | **No** — `UNVERIFIED` (pending local pull and size check) |

## Model Download Instructions (For the user to run)

> [!IMPORTANT]
> Since model files are multi-gigabyte downloads, you should run these Ollama commands manually on your systems.

### 1. Workstation Setup (Local PC)
Run these commands on the workstation to pull all three tiers:
```powershell
ollama pull llama3.2:1b
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
```

### 2. Edge Setup (Raspberry Pi 5)
Run these commands on the Raspberry Pi 5:
```bash
ollama pull llama3.2:1b
ollama pull llama3.1:8b
```
*(Note: `qwen2.5:14b` is excluded from the Pi 5 because it exceeds the 8GB RAM capacity).*

## Verification Steps (After Pulling)

Once pulled, you can verify their sizes and quantizations by running:
```powershell
ollama show --modelfile llama3.2:1b
ollama show --modelfile llama3.1:8b
ollama show --modelfile qwen2.5:14b
```
Please paste the output of these commands or confirm completion to mark them as `Verified` in this manifest.
