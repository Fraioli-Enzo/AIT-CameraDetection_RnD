from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2
import os

def run_inference(image_path, model_version_epoch="25"):
    # Obtenir le dossier du script
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Construire le chemin vers le modèle
    model_path = os.path.join(base_path, f"best{model_version_epoch}.torchscript")
    results_path = os.path.join(base_path, "predicts")
    # Le même chemin correct vers le modèle
    model = YOLO(model_path, task='detect')
    
    # La méthode predict est plus simple à utiliser
    results = model.predict(source=image_path, save=True, save_txt=True, project=results_path, imgsz=640)

    # Pour afficher les résultats
    for r in results:
        img = r.plot()
        cv2.imshow("TEST", img)
        cv2.waitKey(0)

    print(f"Results saved to YOLO_V2 folder")


if __name__ == "__main__":
    model_version_epoch = input("Enter the model version epoch (default is 25): ")
    
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()

    test_image = filedialog.askopenfilename()

    if os.path.exists(test_image):
        run_inference(test_image, model_version_epoch)
        print("Inference completed.")
    else:
        print(f"Test image {test_image} not found.")

