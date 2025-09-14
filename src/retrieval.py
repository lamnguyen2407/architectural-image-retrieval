import numpy as np
from PIL import Image
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

embedding_function = OpenCLIPEmbeddingFunction()

def get_single_image_embedding(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)

# Create client
chroma_client = chromadb.Client()

def create_collection(name="l2_collection", space="l2"):
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"HNSW_SPACE": space}
    )

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

def search(image_path, collection, n_results=5):
    query_image = Image.open(image_path)
    query_embedding = get_single_image_embedding(query_image)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results
