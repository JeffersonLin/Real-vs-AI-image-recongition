# training file for the knn model

from data.data_reader import load_all_data, load_dataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def run_knn():
    data = load_all_data()

    dalle_real = data["dalle"]["real"]
    dalle_fake = data["dalle"]["fake"]

    print("Dalle REAL images:", len(dalle_real))
    print("Dalle FAKE images:", len(dalle_fake))


def sample_images(n):

    all_data = load_all_data()
    datasets = ['dalle','glide','imagen','sd']
    all_samples = {}

    for i in datasets:
        real = all_data[i]['real'] #call real 1 
        fake = all_data[i]['fake'] #call fake 0
        real_sample = real[:n]
        fake_sample = fake[:n]

        for path in real_sample:
            all_samples[path] =1
        
        for path in fake_sample:
            all_samples[path] = 0
    
    return all_samples

def apply_pca(all_samples, size, n_components):
    inputs = []
    labels = []
    for path, label in all_samples.items():
        img = Image.open(path).convert("L") #gray scale according to my notes below
        img = img.resize(size) #512x512
        arr = np.asarray(img) / 255.0
        inputs.append(arr.flatten()) 
        labels.append(label)
    x = np.vstack(inputs) #fpr 2d
    y = np.array(labels, dtype = int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca_model = PCA(n_components)
    x_pca = pca_model.fit_transform(x_scaled)
    return x_pca, y, scaler, pca_model


if __name__ == "__main__":
    samples = sample_images(250)
    # for i in list(samples.keys())[:2000]:
    #     print(i, samples[i])
    x_pca, y, scaler, pca_model = apply_pca(samples, size = (512,512), n_components=50)
    print("PCA shape:", x_pca.shape)
    print("y shape:", y.shape)

#dictionary to classify what dataset it came from
# do a pca transformation channel by channel. so we treat each column channel as an independent gary scale image.