# training file for the knn model

from data.data_reader import load_all_data, load_dataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_knn():
    data = load_all_data()

    dalle_real = data["dalle"]["real"]
    dalle_fake = data["dalle"]["fake"]

    print("Dalle REAL images:", len(dalle_real))
    print("Dalle FAKE images:", len(dalle_fake))

if __name__ == "__main__":
    run_knn()