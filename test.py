import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from tkinter import filedialog
import tkinter as tk

# Fonction pour charger les images et extraire leurs caractéristiques
def extract_features(image_paths):
    # Charger un modèle pré-entraîné (ResNet-18)
    model = models.resnet18(pretrained=True)
    # Supprimer la dernière couche (classification)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    # Transformation pour préparer les images
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    file_names = []
    
    for image_path in image_paths:
        # Charger et prétraiter l'image
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0)
        
        # Extraire les caractéristiques
        with torch.no_grad():
            feature = model(image_tensor)
        
        # Convertir en vecteur 1D
        feature = feature.squeeze().flatten().numpy()
        features.append(feature)
        file_names.append(os.path.basename(image_path))
    
    return np.array(features), file_names

# Clustering des images
def cluster_images(image_paths, n_clusters=2):
    features, file_names = extract_features(image_paths)
    
    # Appliquer K-means pour regrouper les images
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(features)
    
    # Organiser les résultats
    groups = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in groups:
            groups[cluster_id] = []
        groups[cluster_id].append(file_names[i])
    
    return groups

# Exemple d'utilisation avec sélection de fichiers via une boîte de dialogue

# Initialiser tkinter mais cacher la fenêtre principale
root = tk.Tk()
root.withdraw()

# Ouvrir la boîte de dialogue pour sélectionner plusieurs fichiers
image_paths = filedialog.askopenfilenames(
    title="Select Image Files",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
        ("All files", "*.*")
    ]
)

# Regrouper en 2 clusters (vous pouvez ajuster ce nombre)
groups = cluster_images(image_paths, n_clusters=2)

# Afficher les groupes
for cluster_id, images in groups.items():
    print(f"Groupe {cluster_id}:")
    for img in images:
        print(f"  - {img}")