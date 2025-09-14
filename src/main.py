import os, glob, random
from dataset_prep import prepare_dataset
from utils import get_files_path, plot_results
from retrieval import create_collection, add_embedding, search

ROOT = "data"
TRAIN_PATH = f"{ROOT}/train"

def main():
    src_zip = "archive.zip"
    extract_folder = "dataset_raw"

    prepare_dataset(src_zip, extract_folder, TRAIN_PATH)

    class_names = sorted(os.listdir(TRAIN_PATH))
    print("There are", len(class_names), "classes.")

    # Get entire images
    files_path = get_files_path(path=TRAIN_PATH, class_names=class_names)

    # Create collection
    l2_collection = create_collection(name="l2_collection", space="l2")
    add_embedding(l2_collection, files_path)

    # Test query
    random_class = random.choice(class_names)
    test_files = glob.glob(f"{TRAIN_PATH}/{random_class}/*.*")
    test_path = random.choice(test_files)
    print("Choosen class:", random_class)
    print("Testing image:", test_path)

    results = search(test_path, l2_collection, n_results=5)
    plot_results(test_path, results)

if __name__ == "__main__":
    main()
