import cv2
import numpy as np
import ezdxf

# Fonction pour enregistrer les coordonnées au format DXF
def save_to_dxf(contours, filename="output.dxf"):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    
    for contour in contours:
        # Ajouter les lignes des contours
        for i in range(len(contour)):
            start_point = contour[i][0]
            end_point = contour[(i + 1) % len(contour)][0]
            msp.add_line(start=start_point, end=end_point)
        
        # Ajouter les points d'intérêt
        for point in contour:
            msp.add_circle(center=point[0], radius=0.1, dxfattribs={'color': 1})
    
    doc.saveas(filename)
    print(f"Coordonnées enregistrées dans {filename}")

# Ouvrir la connexion à la caméra
cap = cv2.VideoCapture(0)

# Vérifier si la caméra s'est ouverte correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
else:
    print("Appuyez sur 'q' pour quitter la fenêtre vidéo.")
    print("Appuyez sur 's' pour enregistrer les coordonnées au format DXF.")

    while True:
        # Capturer les images frame par frame
        ret, frame = cap.read()

        # Si le frame est lu correctement, ret est True
        if not ret:
            print("Erreur : Impossible de lire le frame.")
            break

        # Définir la région d'intérêt (ROI) sous forme de carré centré
        frame_height, frame_width = frame.shape[:2]
        roi_size = 200  # Taille du carré
        roi_start_x = frame_width // 2 - roi_size // 2
        roi_start_y = frame_height // 2 - roi_size // 2
        roi_end_x = roi_start_x + roi_size
        roi_end_y = roi_start_y + roi_size

        # Extraire la ROI du frame
        roi = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

        # Convertir la ROI en niveaux de gris
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Appliquer le seuillage adaptatif
        thresh = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 16)

        # Détection de contours avec Canny
        edges = cv2.Canny(gray_frame, 100, 200)

        # Transformation morphologique pour détecter les blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Trouver les contours des blobs détectés
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dessiner les contours et les points d'intérêt directement sur l'image originale
        for contour in contours:
            # Calculer l'approximation polygonale du contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Dessiner les contours sur l'image originale
            cv2.drawContours(frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x], [approx], -1, (0, 255, 0), 2)
            
            # Dessiner les points d'intérêt aux sommets du polygone
            for point in approx:
                x, y = point[0]
                cv2.circle(frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x], (x, y), 4, (0, 0, 255), -1)  # Points rouges

        # Afficher le cadre original avec le carré de la ROI et les points détectés
        cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (255, 0, 0), 2)
        cv2.imshow('Cadre Original', frame)

        # Quitter la boucle si 'q' est pressé
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_to_dxf(contours)

    # Libérer la capture lorsque tout est terminé
    cap.release()
    cv2.destroyAllWindows()