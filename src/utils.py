import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)

def folder_to_images(folder, size):
    list_dir = [os.path.join(folder, name) for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    return images_np, np.array(images_path)

def get_files_path(path, class_names):
    files_path = []
    for label in class_names:
        label_path = os.path.join(path, label)
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = os.path.join(label_path, filename)
            files_path.append(filepath)
    return files_path

def plot_results(image_path, results, top_k=5):
    query_img = Image.open(image_path)
    plt.figure(figsize=(15, 3))
    plt.subplot(1, top_k+1, 1)
    plt.imshow(query_img)
    plt.title("Query", fontsize=12)
    plt.axis("off")

    for i, img_path in enumerate(results["ids"][0][:top_k]):
        try:
            img = Image.open(img_path)
            plt.subplot(1, top_k+1, i+2)
            plt.imshow(img)
            plt.title(f"Rank {i+1}", fontsize=12)
            plt.axis("off")
        except Exception as e:
            print(f"Cannot open {img_path}: {e}")

    plt.show()
