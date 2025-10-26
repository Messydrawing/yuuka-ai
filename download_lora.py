from huggingface_hub import snapshot_download
from pathlib import Path

def ensure_models():
    local_dir = Path("models")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Messydrawing/yuuka-ai-models",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False, 
        revision="main",
    )

if __name__ == "__main__":
    ensure_models()
    print("Models ready under ./models")
