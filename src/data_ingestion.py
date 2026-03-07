import pandas as pd
from huggingface_hub import HfApi
import os

# --- CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "cthangella/tourism-dataset" # Rubric: Register on HF dataset space

def ingest_data():
    print("--- [1] Data Ingestion Started ---")

    # Check if local file exists (Upload tourism.csv to Colab Files first!)
    possible_paths = ["tourism.csv", "/content/tourism.csv"]
    file_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not file_path:
        print("Warning: 'tourism.csv' not found locally. Skipping upload.")
        return

    # Rubric: Register the data on the Hugging Face dataset space
    print(f"Uploading {file_path} to {REPO_ID}...")
    try:
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo="raw/tourism.csv",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        print("Data successfully registered on Hugging Face!")
    except Exception as e:
        print(f"Ingestion Failed: {e}")

if __name__ == "__main__":
    ingest_data()
