import os
from huggingface_hub import hf_hub_download

MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_DIR = "models"

def main():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    print(f"Downloading {MODEL_FILENAME} from {MODEL_REPO}...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to {model_path}")

if __name__ == "__main__":
    main()
