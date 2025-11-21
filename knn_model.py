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


def sample_images(n = 250):

    all_data = load_all_data()
    datasets = ['dalle','glide','imagen','sd']
    all_samples = []

    for i in datasets:
        real = all_data[i]['real']
        fake = all_data[i]['fake']
        real_sample = real[:250]
        fake_sample = fake[:250]

        all_samples.append(real_sample)
        all_samples.append(fake_sample)
    
    return all_samples






if __name__ == "__main__":
    samples = sample_images(250)
    print(len(samples))
