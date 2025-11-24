# training file for the knn model

from data.data_reader import load_all_data, load_dataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def run_knn(x, y, n_neighbors=5):
    data = load_all_data()

    dalle_real = data["dalle"]["real"]
    dalle_fake = data["dalle"]["fake"]

    # print("Dalle REAL images:", len(dalle_real))
    # print("Dalle FAKE images:", len(dalle_fake))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", weights="uniform")
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    print("KNN accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    return knn


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
        img = Image.open(path).convert("L")     #gray scale according to my notes below
        img = img.resize(size)                  #512x512
        arr = np.asarray(img) / 255.0           #normalizes
        inputs.append(arr.flatten())            #flattens
        labels.append(label)
    x = np.vstack(inputs) #fpr 2d
    y = np.array(labels, dtype = int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca_model = PCA(n_components)
    x_pca = pca_model.fit_transform(x_scaled)
    
    return x_pca, y, scaler, pca_model

def apply_patch_pca(all_samples, size, patch_size, n_components):
    ph, pw = patch_size

    items = list(all_samples.items())  #(path, label)

    patch_list = []
    patches_per_image = []
    labels = []

    # loop through each img
    for path, label in items:
        img = Image.open(path).convert("L") #gray scale
        img = img.resize(size)
        arr = np.asarray(img) / 255.0 

        H, W = arr.shape
        nrows, ncols = H // ph, W // pw

        arr = arr[:nrows*ph, :ncols*pw]

        count_patches = 0
        for ii in range(nrows):
            for jj in range(ncols):
                block = arr[ii*ph:(ii+1)*ph, jj*pw:(jj+1)*pw]
                vec = block.reshape(-1, order='F')
                patch_list.append(vec)
                count_patches += 1

        patches_per_image.append(count_patches)
        labels.append(label)

    X_patches = np.vstack(patch_list)
    y = np.array(labels, dtype=int)

    scaler = StandardScaler()
    X_patches_scaled = scaler.fit_transform(X_patches)

    pca_model = PCA(n_components=n_components)
    X_patches_pca = pca_model.fit_transform(X_patches_scaled)

    num_images = len(items)
    X_img = np.zeros((num_images, n_components))

    # combine the imgs back together
    idx = 0
    for i in range(num_images):
        m = patches_per_image[i]           # how many patches in this image
        feat_patches = X_patches_pca[idx:idx+m]  # (m, n_components)
        X_img[i, :] = feat_patches.mean(axis=0)
        idx += m

    return X_img, y, scaler, pca_model


if __name__ == "__main__":
    samples = sample_images(250)
    
    x_pca, y, scaler, pca_model = apply_pca(samples, size = (128,128), n_components=50)
    # print("PCA shape:", x_pca.shape)
    # print("y shape:", y.shape)

    x_patch_pca, y, patch_scaler, patch_pca_model = apply_patch_pca(samples, size=(128, 128), patch_size=(8, 8), n_components=50)
    # print("Patch-PCA feature shape:", x_patch_pca.shape)
    # print("Labels shape:", y.shape)

    knn_model = run_knn(x_patch_pca, y, n_neighbors=5)
    knn_model_2 = run_knn(x_pca, y, n_neighbors=5)

#dictionary to classify what dataset it came from
# do a pca transformation channel by channel. so we treat each column channel as an independent gary scale image.