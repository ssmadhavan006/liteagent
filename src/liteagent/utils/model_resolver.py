import os
import json
from huggingface_hub import hf_hub_download

# Mapping of model tags to Hugging Face repositories and files
HF_MAPPINGS = {
    "llama3.2:1b": {
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    },
    "llama3.2:3b": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    },
    "llama3.1:8b": {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    }
}

# Directories to check for Ollama installation
OLLAMA_PATHS = [
    r"D:\Ollama\Models",
    os.path.expanduser(r"~\.ollama\models"),
    r"C:\ProgramData\Ollama"
]

def resolve_model_path(model_tag: str) -> str:
    """
    Resolves the absolute path to a model GGUF file.
    First tries to locate the local Ollama registry manifest and its corresponding blob.
    If that fails, falls back to downloading the GGUF file from Hugging Face.
    """
    # 1. Parse tag to find manifest name (e.g. 'llama3.2:1b' -> 'llama3.2', '1b')
    if ":" in model_tag:
        name, tag = model_tag.split(":", 1)
    else:
        name = model_tag
        tag = "latest"
        
    # Check each possible Ollama installation path
    for base_path in OLLAMA_PATHS:
        if not os.path.exists(base_path):
            continue
            
        manifest_path = os.path.join(base_path, "manifests", "registry.ollama.ai", "library", name, tag)
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                
                # Find layer with model mediaType
                for layer in manifest.get("layers", []):
                    if layer.get("mediaType") == "application/vnd.ollama.image.model":
                        digest = layer.get("digest")
                        if digest and digest.startswith("sha256:"):
                            blob_filename = "sha256-" + digest[7:]
                            blob_path = os.path.join(base_path, "blobs", blob_filename)
                            if os.path.exists(blob_path):
                                return blob_path
            except Exception as e:
                pass
                
    # 2. Fallback to Hugging Face download
    if model_tag not in HF_MAPPINGS:
        raise ValueError(f"Unknown model tag: {model_tag} and no local Ollama manifest found.")
        
    mapping = HF_MAPPINGS[model_tag]
    print(f"Ollama manifest/blob not found for {model_tag}. Falling back to Hugging Face download...")
    
    # Download to standard workspace models cache directory
    local_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        path = hf_hub_download(
            repo_id=mapping["repo_id"],
            filename=mapping["filename"],
            local_dir=local_dir
        )
        return os.path.abspath(path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_tag} from Hugging Face: {e}") from e
