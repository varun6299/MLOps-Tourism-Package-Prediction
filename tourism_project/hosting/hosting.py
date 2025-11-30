from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/content/drive/MyDrive/Masters_2025_AI_ML/MLOps/tourism_project/deployment",     # the local folder containing your files
    repo_id="Varun6299/Tourism-Package-Prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
