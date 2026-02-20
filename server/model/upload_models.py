from huggingface_hub import upload_file

REPO_ID = "iam-i-j-k/agriscan-model"

# You are in backend/model/, so just use local file names
upload_file(
    path_or_fileobj="model.pth",
    path_in_repo="model.pth",
    repo_id=REPO_ID,
    repo_type="model"
)

upload_file(
    path_or_fileobj="leaf_binary_classifier.pth",
    path_in_repo="leaf_binary_classifier.pth",
    repo_id=REPO_ID,
    repo_type="model"
)

upload_file(
    path_or_fileobj="class_names.json",
    path_in_repo="class_names.json",
    repo_id=REPO_ID,
    repo_type="model"
)
