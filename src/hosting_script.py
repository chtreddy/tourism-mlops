from huggingface_hub import HfApi
import os

# Define a hosting script that can push all deployment files to Hugging Face Space
HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO = "cthangella/tourism-app" # Your Space Name

def deploy_to_space():
    print(f"Deploying App to Hugging Face Space: {SPACE_REPO}...")
    try:
        api = HfApi(token=HF_TOKEN)
        # Upload the entire 'app' folder content to the Space
        api.upload_folder(
            folder_path="/content/tourism_project/app",
            repo_id=SPACE_REPO,
            repo_type="space"
        )
        print(" Deployment Successful! App is live.")
    except Exception as e:
        print(f" Deployment Failed: {e}")

if __name__ == "__main__":
    deploy_to_space()
