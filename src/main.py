# src/main.py
# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# Initialize embedding function
embedding_function = OpenCLIPEmbeddingFunction()

# --- Image processing utilities ---
def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)

def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path

def find_images_folder(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if any(fname.lower().endswith(".jpg") for fname in filenames):
            return dirpath
    return None

def get_single_image_embedding(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)

def get_files_path(ROOT, CLASS_NAME):
    files_path = []
    for label in CLASS_NAME:
        label_path = ROOT + "/" + label
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = label_path + '/' + filename
            files_path.append(filepath)
    return files_path

# --- ChromaDB utils ---
def add_embedding(collection, files_path):
    for file_path in tqdm(files_path):
        img = Image.open(file_path).convert("RGB")
        img_np = np.array(img)
        embedding = get_single_image_embedding(img_np)
        collection.add(
            embeddings=[embedding.tolist()],
            documents=[file_path],
            ids=[file_path]
        )

def search(image_path, collection, n_results):
    query_image = Image.open(image_path)
    query_embedding = get_single_image_embedding(query_image)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results
