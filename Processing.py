import ezdxf
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

# 1️⃣ Charger les points depuis un fichier DXF
def extract_points_from_dxf(file_path):
    """Lit un fichier DXF et extrait les points."""
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    points = []

    for entity in msp:
        print(entity.dxftype())  # Afficher le type d'entité

    for entity in msp.query('POINT'):
        point = (entity.dxf.location.x, entity.dxf.location.y)  # Utiliser location
        points.append(point)

    return points

# 2️⃣ Fusionner les points trop proches
def merge_close_points(points, threshold=5):
    """Fusionne les points trop proches en un seul en prenant la moyenne."""
    if not points:
        return []

    tree = KDTree(points)
    clusters = []
    visited = set()

    for i, point in enumerate(points):
        if i in visited:
            continue
        indices = tree.query_ball_point(point, threshold)
        if len(indices) > 1:  # Fusionner seulement si plus d'un point est trouvé
            cluster = np.mean([points[j] for j in indices], axis=0)  # Moyenne des points du cluster
            clusters.append(tuple(cluster))
        else:
            clusters.append(point)  # Garder le point original s'il est seul
        visited.update(indices)

    return clusters


# 3️⃣ Trier les points pour former un contour logique
def order_points(points):
    """Trie les points pour former un contour logique."""
    if not points:
        return []

    points = points[:]  # Copie des points pour ne pas modifier l'original
    ordered = [points.pop(0)]  # On commence par le premier point
    while points:
        last_point = ordered[-1]
        distances = cdist([last_point], points)  # Calcul des distances aux autres points
        nearest_index = np.argmin(distances)  # Trouver le plus proche
        ordered.append(points.pop(nearest_index))

    return ordered

# 4️⃣ Enregistrer le polygone fermé dans un fichier DXF avec un effet miroir de bas en haut
def save_polygon_to_dxf(points, output_path):
    """Crée un fichier DXF avec un polygone fermé à partir des points traités avec un effet miroir de bas en haut."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    if points:
        # Appliquer l'effet miroir de bas en haut
        mirrored_points = [(x, -y) for x, y in points]
        mirrored_points.append(mirrored_points[0])  # Fermer le polygone
        msp.add_lwpolyline(mirrored_points, close=True)  # Ajouter un polygone fermé

    doc.saveas(output_path)

def calculate_polygon_area(points, unit='unit'):
    """Calcule l'aire d'un polygone à partir des points ordonnés et affiche l'unité."""
    if len(points) < 3:  # Un polygone doit avoir au moins 3 points
        return 0

    area = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]  # Le point suivant, en bouclant au début
        area += x1 * y2 - y1 * x2

    area = abs(area) / 2
    area_cm2 = area / 100  # Convertir l'aire de mm^2 en cm^2
    print(f"🔲 {area_cm2:.1f} aire du polygone en cm^2")
    return area

# Utilisation dans le pipeline complet
def process_dxf(input_dxf, output_dxf, threshold=5, unit='unit'):
    """Exécute le pipeline complet de traitement du fichier DXF."""
    print("📂 Lecture du fichier DXF...")
    points = extract_points_from_dxf(input_dxf)

    print(f"🔍 {len(points)} points extraits.")
    
    print("📏 Fusion des points trop proches...")
    filtered_points = merge_close_points(points, threshold)
    print(f"✅ {len(filtered_points)} points après fusion.")

    print("📌 Tri des points pour former un contour...")
    ordered_points = order_points(filtered_points)

    calculate_polygon_area(ordered_points, unit)

    print("💾 Sauvegarde du polygone dans un nouveau DXF...")
    save_polygon_to_dxf(ordered_points, output_dxf)
    
    print(f"🎉 Polygone enregistré dans {output_dxf}")

# Exécution
input_dxf = "output.dxf"  # Remplace par ton fichier d'entrée
output_dxf = "output2.dxf"  # Nom du fichier de sortie
unit = 'mm'  # Unité des coordonnées des points dans le fichier DXF

process_dxf(input_dxf, output_dxf, threshold=5, unit=unit)
