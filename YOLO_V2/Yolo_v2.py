from ultralytics import YOLO
import torch
import os

def train_model():
    model = YOLO("yolov8n.pt")  # Load YOLOv8 model
    model.train(data="D:\Enzo\datasets\FabricDefects2\data.yaml", epochs=25, batch=16, imgsz=416, project="YOLO_V2")

def evaluate_model():
    # Le chemin correct vers le modèle entraîné
    model = YOLO("YOLO_V2/train/weights/best.pt")
    metrics = model.val(project="YOLO_V2")
    print(metrics)

def run_inference(image_path):
    # Le même chemin correct vers le modèle
    model = YOLO("YOLO_V2/train/weights/best.pt")
    # La méthode predict est plus simple à utiliser
    results = model.predict(source=image_path, save=True, save_txt=True, project="YOLO_V2/predicts")
    
    # Pour afficher les résultats
    import cv2
    for r in results:
        img = r.plot()
        cv2.imshow("Detection Result", img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print(f"Results saved to YOLO_V2 folder")

def export_model():
    model = YOLO("YOLO_V2/train/weights/best.pt")
    model.export(format="onnx")  # Export to ONNX

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("Training on CPU. Consider using GPU for better performance.")
    
    train_model()
    print("Training completed.")
    
    # Vérifiez si le fichier de modèle existe avant de continuer
    if os.path.exists("YOLO_V2/train/weights/best.pt"):
        evaluate_model()
        print("Evaluation completed.")
        
        export_model()
        print("Model exported.")
    else:
        print("ERROR: Le fichier de modèle entraîné n'a pas été trouvé.")
        print("Chemin attendu: YOLO_V2/train/weights/best.pt")
        print("Vérifiez le dossier de sortie de l'entraînement et ajustez les chemins.")
