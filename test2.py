import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

# Variables globales pour stocker les états entre les fonctions
img1 = None
img2 = None
dominant_colors = None
result_img = None
threshold = 100  # Seuil de tolérance pour la correspondance des couleurs (0-255)

def load_images(img1_path, img2_path):
    """Charge les deux images depuis les chemins spécifiés."""
    global img1, img2
    
    # Vérifier que les fichiers existent
    if not os.path.exists(img1_path):
        print(f"Erreur: L'image source {img1_path} n'existe pas.")
        return False
        
    if not os.path.exists(img2_path):
        print(f"Erreur: L'image cible {img2_path} n'existe pas.")
        return False
    
    # Charger les images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Erreur lors du chargement des images.")
        return False
    
    # Convertir de BGR à RGB pour l'affichage avec matplotlib
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    print(f"Images chargées avec succès:")
    print(f"Image source: {img1_path} - Dimensions: {img1.shape}")
    print(f"Image cible: {img2_path} - Dimensions: {img2.shape}")
    
    return True
    
def extract_dominant_colors(n_colors=5):
    """Extrait les couleurs dominantes de la première image."""
    global dominant_colors, img1
    
    # Redimensionner l'image pour accélérer le traitement
    resized_img = cv2.resize(img1, (100, 100))
    
    # Reformater l'image pour KMeans
    pixels = resized_img.reshape(-1, 3)
    
    # Appliquer KMeans pour trouver les couleurs dominantes
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Récupérer les centres (= couleurs dominantes)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    print(f"Couleurs dominantes extraites: {dominant_colors}")
    return dominant_colors
    
def highlight_colors(brightness_factor=1.5, saturation_factor=1.2):
    """Met en surbrillance les pixels de la deuxième image qui correspondent aux couleurs dominantes."""
    global img2, dominant_colors, result_img, threshold
    
    if img2 is None or dominant_colors is None:
        print("Erreur: Images ou couleurs non chargées.")
        return None
        
    # Créer une copie de l'image cible
    result_img = img2.copy()
    
    # Convertir l'image en HSV pour manipuler la luminosité et la saturation
    img_hsv = cv2.cvtColor(result_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Créer un masque pour les pixels à mettre en surbrillance
    mask = np.zeros(img2.shape[:2], dtype=bool)
    
    # Pour chaque couleur dominante
    for color in dominant_colors:
        # Créer un masque pour les pixels proches de cette couleur
        color_mask = np.all(np.abs(img2 - color) < threshold, axis=2)
        mask = mask | color_mask
    
    # Augmenter la luminosité et la saturation des pixels correspondants
    img_hsv[mask, 1] = np.clip(img_hsv[mask, 1] * saturation_factor, 0, 255)  # Saturation
    img_hsv[mask, 2] = np.clip(img_hsv[mask, 2] * brightness_factor, 0, 255)  # Luminosité
    
    # Reconvertir en RGB
    result_img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    print(f"Mise en surbrillance effectuée avec un seuil de {threshold}")
    return result_img

def adjust_threshold(new_threshold):
    """Ajuste le seuil de tolérance pour la correspondance des couleurs."""
    global threshold
    threshold = new_threshold
    highlight_colors()
    show_results()

def save_result(output_path):
    """Sauvegarde l'image résultante."""
    global result_img
    
    if result_img is None:
        print("Erreur: Aucune image résultante à sauvegarder.")
        return False
    
    # Convertir en BGR pour OpenCV
    save_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, save_img)
    print(f"Image sauvegardée: {output_path}")
    return True

def show_results():
    """Affiche les résultats: image source, couleurs extraites, image cible, résultat."""
    global img1, img2, dominant_colors, result_img, threshold
    
    plt.figure(figsize=(15, 10))
    
    # Image source
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title('Image source')
    plt.axis('off')
    
    # Palettes de couleurs extraites
    plt.subplot(2, 2, 2)
    color_display = np.ones((100, 500, 3), dtype=np.uint8) * 255
    
    n_colors = len(dominant_colors)
    width = 500 // n_colors
    
    for i, color in enumerate(dominant_colors):
        start_x = i * width
        end_x = (i + 1) * width
        color_display[:, start_x:end_x] = color
        
        # Ajouter le code RGB
        rgb_text = f"RGB: {color[0]},{color[1]},{color[2]}"
        plt.text(start_x + width//2, 80, rgb_text, 
                 ha='center', va='center', 
                 color='black' if np.mean(color) > 128 else 'white')
    
    plt.imshow(color_display)
    plt.title('Couleurs dominantes extraites')
    plt.axis('off')
    
    # Image cible
    plt.subplot(2, 2, 3)
    plt.imshow(img2)
    plt.title('Image cible')
    plt.axis('off')
    
    # Résultat
    plt.subplot(2, 2, 4)
    plt.imshow(result_img)
    plt.title('Résultat avec surbrillance')
    plt.axis('off')
    
    # Ajouter des boutons pour ajuster le seuil
    plt.subplots_adjust(bottom=0.2)
    
    ax_decrease = plt.axes([0.3, 0.05, 0.15, 0.075])
    ax_increase = plt.axes([0.55, 0.05, 0.15, 0.075])
    
    btn_decrease = Button(ax_decrease, f'-10 (Seuil: {threshold})')
    btn_increase = Button(ax_increase, f'+10 (Seuil: {threshold})')
    
    def decrease_threshold(event):
        new_threshold = max(5, threshold - 10)
        print(f"Diminution du seuil: {threshold} -> {new_threshold}")
        adjust_threshold(new_threshold)
        plt.close()
        
    def increase_threshold(event):
        new_threshold = min(255, threshold + 10)
        print(f"Augmentation du seuil: {threshold} -> {new_threshold}")
        adjust_threshold(new_threshold)
        plt.close()
        
    btn_decrease.on_clicked(decrease_threshold)
    btn_increase.on_clicked(increase_threshold)
    
    # Ajouter un bouton pour sauvegarder
    ax_save = plt.axes([0.7, 0.05, 0.15, 0.075])
    btn_save = Button(ax_save, 'Sauvegarder')
    
    def save_image(event):
        output_path = input("Entrez le chemin pour sauvegarder l'image (ex: resultat.png): ")
        save_result(output_path)
            
    btn_save.on_clicked(save_image)
    
    plt.tight_layout()
    plt.show()

def run(img1_path, img2_path, n_colors=5):
    """Exécute le programme complet."""
    if not load_images(img1_path, img2_path):
        return
        
    extract_dominant_colors(n_colors)
    highlight_colors()
    show_results()

# Programme principal
if __name__ == "__main__":
    print("Programme de mise en surbrillance des couleurs")
    print("=============================================")
    
    # Demander les chemins des images directement
    img1_path = input("Chemin de l'image source (pour extraire les couleurs): ")
    img2_path = input("Chemin de l'image cible (pour mettre en surbrillance): ")
    
    # Demander le nombre de couleurs à extraire
    try:
        n_colors = int(input("Nombre de couleurs dominantes à extraire (par défaut: 5): ") or "5")
    except ValueError:
        print("Valeur invalide, utilisation de la valeur par défaut (5)")
        n_colors = 5
    
    run(img1_path, img2_path, n_colors)