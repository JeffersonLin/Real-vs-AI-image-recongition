# training file for the svm model

from data.data_reader import load_all_data, load_dataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def run_svm(x, y, kernel, C=1.0, gamma='scale'):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)

    print(f"\nSVM ({kernel} kernel) results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    return svm


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

def tune_svm(X, y, kernel='rbf'):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01, 0.001]
    }
    svm = SVC(kernel=kernel)
    grid = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    return grid.best_estimator_


if __name__ == "__main__":
    samples = sample_images(250)
    
    x_pca, y, scaler, pca_model = apply_pca(samples, size = (128,128), n_components=50)
    # print("PCA shape:", x_pca.shape)
    # print("y shape:", y.shape)
   
    x_patch_pca, y_patch, patch_scaler, patch_pca_model = apply_patch_pca(samples, size=(128, 128), patch_size=(8, 8), n_components=50)
    # print("Patch-PCA feature shape:", x_patch_pca.shape)
    # print("Labels shape:", y.shape)

    ## check for y being overwritten in other code!!! fix: y -> y_patch for patch pca ##

    # # Grid search SVM tuning
    # best_svm_rbf = tune_svm(x_pca, y, kernel='rbf')
    # best_svm_rbf = tune_svm(x_patch_pca, y, kernel='rbf')

    # best svm
    svm_rbf_tuned = run_svm(x_patch_pca, y, kernel='rbf', C=10.0, gamma=0.001)
    # svm_lin_tuned = run_svm(x_pca, y, kernel='linear', C=1.0, gamma='scale')
    # svm_lin_tuned = run_svm(x_patch_pca, y, kernel='linear', C=1.0, gamma='scale')