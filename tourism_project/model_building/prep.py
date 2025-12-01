# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Loading Data From HuggingFace Dataset Space
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Varun6299/Tourism-Package-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")


# Data Cleaning & Removing Unwanted Columns

## Dropping CustomerID
df.drop(['Unnamed: 0', 'CustomerID'],axis=1, inplace=True)


numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'PitchSatisfactionScore',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation',
    'Passport',
    'OwnCar'
]

## Data Cleaning
df.Gender.replace({"Fe Male":"Female"},inplace=True)

df.Occupation.replace({"Free Lancer":"Freelancer"},inplace=True)

df.MaritalStatus.replace({"Unmarried":"Single"},inplace=True)

target_col = 'ProdTaken'

# Split Cleaned Data Into into Train & Test Sets
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]


# Uploading Train & Test Sets Back To Hugging
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Varun6299/Tourism-Package-Prediction",
        repo_type="dataset",
    )
