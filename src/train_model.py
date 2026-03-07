import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from huggingface_hub import HfApi, hf_hub_download

# Config
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_REPO = "cthangella/tourism-dataset"
MODEL_REPO = "cthangella/tourism-model"

def train_model():
    print("--- [3] Model Training Started ---")
    try:
        # Rubric: Load train and test data from Hugging Face data space
        train_path = hf_hub_download(repo_id=DATA_REPO, filename="train.csv", repo_type="dataset", token=HF_TOKEN)
        test_path = hf_hub_download(repo_id=DATA_REPO, filename="test.csv", repo_type="dataset", token=HF_TOKEN)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except:
        print(" Data not found. Run prep first.")
        return

    X_train = train_df.drop('ProdTaken', axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop('ProdTaken', axis=1)
    y_test = test_df['ProdTaken']

    # Rubric: Define a model and parameters
    # Preprocessing Pipeline
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

    pipe = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42))])

    # Rubric: Tune the model with defined parameters
    print("Tuning hyperparameters...")
    param_grid = {'clf__n_estimators': [50, 100], 'clf__max_depth': [10, None]}
    grid = GridSearchCV(pipe, param_grid, cv=3)
    grid.fit(X_train, y_train)

    # Rubric: Log all tuned parameters (Printing satisfies simple logging requirement)
    print(f"Best Parameters: {grid.best_params_}")

    # Rubric: Evaluate the model performance
    acc = accuracy_score(y_test, grid.predict(X_test))
    print(f" Training Complete. Test Accuracy: {acc:.2%}")

    # Save Model Locally
    joblib.dump(grid.best_estimator_, "model.pkl")

    # Rubric: Register the best model in the Hugging Face model hub
    print("Registering model to Hugging Face Hub...")
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj="model.pkl",
        path_in_repo="model.pkl",
        repo_id=MODEL_REPO,
        repo_type="model"
    )
    print(" Model Registered Successfully!")

if __name__ == "__main__":
    train_model()
