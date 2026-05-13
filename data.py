import os

split_path = r"C:\Users\rajes\Desktop\Project\dataset_split\train"

for cls in os.listdir(split_path):
    n = len(os.listdir(os.path.join(split_path, cls)))
    print(f"{cls}: {n} images")