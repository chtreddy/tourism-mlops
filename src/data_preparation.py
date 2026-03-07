import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "cthangella/tourism-dataset"

def prepare_data():
    print("--- [2] Data Preparation Started ---")
    try:
        # Load dataset directly from Hugging Face data space
        print("Loading data from Hugging Face...")
        path = hf_hub_download(repo_id=REPO_ID, filename="raw/tourism.csv", repo_type="dataset", token=HF_TOKEN)
        df = pd.read_csv(path)

        # Perform data cleaning and remove unnecessary columns
        print("Cleaning data...")
        if 'Gender' in df.columns: df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
        if 'DurationOfPitch' in df.columns: df['DurationOfPitch'] = df['DurationOfPitch'].fillna(df['DurationOfPitch'].median())
        if 'TypeofContact' in df.columns: df['TypeofContact'] = df['TypeofContact'].fillna('Self Enquiry')

        # Remove IDs
        df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore', inplace=True)

        # Split cleaned dataset into train/test and save locally
        print("Splitting data...")
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train.to_csv("train.csv", index=False)
        test.to_csv("test.csv", index=False)

        #Upload resulting train/test datasets back to HF data space
        print("Uploading processed data to Hugging Face...")
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj="train.csv", path_in_repo="train.csv", repo_id=REPO_ID, repo_type="dataset")
        api.upload_file(path_or_fileobj="test.csv", path_in_repo="test.csv", repo_id=REPO_ID, repo_type="dataset")
        print(" Data Prepared & Uploaded!")

    except Exception as e:
        print(f" Preparation Error: {e}")

if __name__ == "__main__":
    prepare_data()
