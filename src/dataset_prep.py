import os
import zipfile
import shutil

# Find the subfolder containing JPG images
def find_images_folder(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if any(fname.lower().endswith(".jpg") for fname in filenames):
            return dirpath
    return None

# Extract and classify dataset
def prepare_dataset(src_zip, extract_folder, dst_folder):
    # Extract zip file
    with zipfile.ZipFile(src_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print("Extracted at:", extract_folder)

    src_folder = find_images_folder(extract_folder)
    print("Images stored at:", src_folder)

    # Classify images by buildingXX
    for filename in os.listdir(src_folder):
        if filename.lower().endswith(".jpg"):
            class_name = filename.split("_")[0]
            class_folder = os.path.join(dst_folder, class_name)
            os.makedirs(class_folder, exist_ok=True)

            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(class_folder, filename)
            shutil.copy(src_path, dst_path)

    print("Classification completed")

    return dst_folder
