import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.fftpack import fft2, fftshift
from tkinter import filedialog
import tkinter as tk

def preprocess_image(image_path, size=(256, 256)):
    """Prétraiter l'image: redimensionner et convertir en niveaux de gris."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def compute_texture_features(gray_img):
    """Calculer les caractéristiques de texture invariantes à la rotation."""
    # LBP (Local Binary Pattern) avec paramètres pour l'invariance à la rotation
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    
    # Histogramme LBP
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    # GLCM (Gray-Level Co-Occurrence Matrix) pour les propriétés de texture
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Multiples angles pour l'invariance
    glcm = graycomatrix(gray_img, distances, angles, 256, symmetric=True, normed=True)
    
    # Extraire diverses propriétés de la GLCM
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # Analyse de Fourier pour détecter les patterns périodiques (carreaux, lignes)
    f_transform = fftshift(np.abs(fft2(gray_img)))
    f_transform_log = np.log1p(f_transform)
    
    # Moyenne radiale du spectre pour l'invariance à la rotation
    h, w = f_transform_log.shape
    center = (w//2, h//2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Créer des bins pour la moyenne radiale
    max_radius = min(center[0], center[1])
    num_bins = 50
    radial_bins = np.linspace(0, max_radius, num_bins+1)
    radial_profile = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (dist_from_center >= radial_bins[i]) & (dist_from_center < radial_bins[i+1])
        if mask.any():
            radial_profile[i] = f_transform_log[mask].mean()
    
    # Normaliser le profil radial
    radial_profile = radial_profile / (np.sum(radial_profile) + 1e-10)
    
    # Combiner toutes les caractéristiques
    features = np.concatenate([
        hist,
        [contrast, dissimilarity, homogeneity, energy, correlation],
        radial_profile
    ])
    
    return features

def detect_anomalies(image_paths, threshold=0.8):
    """
    Grouper les images entre normales et anormales.
    
    Args:
        image_paths: Liste des chemins d'images
        threshold: Seuil de similarité (0-1) pour considérer deux images comme similaires
        
    Returns:
        dict: Dictionnaire avec deux clés "normal" et "anomalies" contenant les chemins d'images
    """
    features_list = []
    
    # Extraire les caractéristiques de chaque image
    for path in image_paths:
        _, gray = preprocess_image(path)
        features = compute_texture_features(gray)
        features_list.append(features)
    
    # Calculer la matrice de similarité
    features_array = np.array(features_list)
    similarity_matrix = cosine_similarity(features_array)
    
    # Trouver l'image la plus représentative (celle qui est la plus similaire aux autres)
    avg_similarities = np.mean(similarity_matrix, axis=1)
    most_representative_idx = np.argmax(avg_similarities)
    
    # Comparer toutes les images à l'image la plus représentative
    normal_images = []
    anomaly_images = []
    
    for i, path in enumerate(image_paths):
        if similarity_matrix[most_representative_idx, i] >= threshold:
            normal_images.append(path)
        else:
            anomaly_images.append(path)
    
    return {
        "normal": normal_images,
        "anomalies": anomaly_images
    }

def visualize_results(image_paths, groups):
    """Visualiser les images normales et anormales."""
    n_normal = len(groups["normal"])
    n_anomalies = len(groups["anomalies"])
    
    # Déterminer le nombre de lignes et colonnes
    cols = min(5, max(n_normal, n_anomalies))
    rows = ((n_normal + cols - 1) // cols) + ((n_anomalies + cols - 1) // cols)
    
    fig = plt.figure(figsize=(15, 3 * rows))
    
    # Afficher les images normales
    for i, path in enumerate(groups["normal"]):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"Normal: {path.split('/')[-1]}")
        ax.axis('off')
    
    # Afficher les images anormales
    start_idx = n_normal + 1
    for i, path in enumerate(groups["anomalies"]):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(rows, cols, start_idx + i)
        ax.imshow(img)
        ax.set_title(f"Anomalie: {path.split('/')[-1]}", color='red')
        ax.axis('off')
    
    plt.tight_layout()
# Exemple d'utilisation
if __name__ == "__main__":
    # Liste des chemins vers vos images de tissu
    # Initialiser Tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    

    # Ouvrir la boîte de dialogue pour sélectionner plusieurs fichiers
    image_paths = filedialog.askopenfilenames(
        title="Select Image Files",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    # Détecter les anomalies (vous pouvez ajuster le seuil)
    groups = detect_anomalies(image_paths, threshold=0.75)
    
    # Afficher les résultats
    print("Images normales:", len(groups["normal"]))
    for img in groups["normal"]:
        print(f"  - {img}")
    
    print("\nImages avec anomalies:", len(groups["anomalies"]))
    for img in groups["anomalies"]:
        print(f"  - {img}")
    
    # Visualiser les résultats
    visualize_results(image_paths, groups)