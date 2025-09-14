# Content-Based Image Retrieval using CLIP + ChromaDB

This project implements a **Content-Based Image Retrieval (CBIR)** system for architectural image datasets using [OpenCLIP](https://github.com/mlfoundations/open_clip) embeddings and [ChromaDB](https://www.trychroma.com/) as a vector database.

---

## 🚀 Features
- **Dataset preparation**: extract & organize dataset into class folders.
- **Image embeddings**: use OpenCLIP to encode images into vectors.
- **Vector database**: store embeddings in ChromaDB for efficient similarity search.
- **Retrieval metrics**: search with multiple distance functions (L1, L2, cosine, correlation).
- **Visualization**: display query vs. top-k retrieved results.

---

## ⚙️ Installation
Clone this repo and install dependencies:
git clone https://github.com/<your-username>/image-retrieval-clip.git
cd image-retrieval-clip
pip install -r requirements.txt

---
## 📊 Usage
**Run pipeline locally:**
python main.py

**Run on Google Colab:**

1. Open notebooks/colab_notebook.ipynb.
2. Upload your dataset zip (e.g., archive.zip).
3. Run all cells.

---
## 📸 Example Output

Input: query image from dataset

Output: top-5 most similar images retrieved from the collection

The system will display something like:

[Query Image] [Rank 1] [Rank 2] [Rank 3] [Rank 4] [Rank 5]

---
## 📦 Dependencies

* numpy
* pillow
* matplotlib
* tqdm
* chromadb
* open-clip-torch
* kagglehub

All included in requirements.txt.
