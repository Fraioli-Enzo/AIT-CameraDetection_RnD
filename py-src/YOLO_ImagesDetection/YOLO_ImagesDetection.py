from ultralytics import YOLO
import torch
import cv2
import os

def run_inference(image_path):
    # Le même chemin correct vers le modèle
    model = YOLO("YOLO_V2/train3/weights/best.torchscript", task='detect')
    # La méthode predict est plus simple à utiliser

    results = model.predict(source=image_path, save=True, save_txt=True, project="YOLO_V2/predicts", imgsz=640)

    # Pour afficher les résultats
    for r in results:
        img = r.plot()
        cv2.imshow("TEST", img)
        cv2.waitKey(0)

    print(f"Results saved to YOLO_V2 folder")


if __name__ == "__main__":
    test_image = "Images/anomali_15.png"  # Replace with an actual test image
    if os.path.exists(test_image):
        run_inference(test_image)
        print("Inference completed.")
    else:
        print(f"Test image {test_image} not found.")

