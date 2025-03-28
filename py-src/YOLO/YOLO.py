from ultralytics import YOLO
import torch
import os
'''
/!\ IMPORTANT /!\
Changes all paths to the correct ones before running the script.
'''
def train_model():
    model = YOLO("yolo11n.pt")  # Load YOLOv8 model
    model.train(data="C:/Users/user/Documents/ENZO/TEST_TEST/datasets/FDDv2/data.yaml", epochs=100, batch=32, imgsz=640, project="YOLO_V2")

def evaluate_model():
    # Le chemin correct vers le modèle entraîné
    model = YOLO("YOLO_V2/train5/weights/best.pt")
    metrics = model.val(project="YOLO_V2")
    print(metrics)

def export_model():
    model = YOLO("YOLO_V2/train5/weights/best.pt")
    model.export(format="torchscript") 


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("####################################")
        print("CUDA is available. Training on GPU.")
        print("####################################")
    else:
        print("----------------------/!\CPU/!\----------------------")

    if not os.path.exists("YOLO_V2"):
        os.makedirs("YOLO_V2")

    train_model()
    print("Training completed.")
    
    # Vérifiez si le fichier de modèle existe avant de continuer
    if os.path.exists("YOLO_V2/train5/weights/best.pt"):
        evaluate_model()
        print("Evaluation completed.")
        
        export_model()
        print("Model exported.")
    else:
        print("ERROR: Le fichier de modèle entraîné n'a pas été trouvé.")
        print("Chemin attendu: YOLO_V2/train5/weights/best.pt")
        print("Vérifiez le dossier de sortie de l'entraînement et ajustez les chemins.")


# print("CUDA version de PyTorch :", torch.version.cuda)  
# print("CUDA disponible :", torch.cuda.is_available())  
# print("Nombre de GPU détectés :", torch.cuda.device_count())  

# if torch.cuda.is_available():
#     print("Nom du GPU :", torch.cuda.get_device_name(0))  
# else:
#     print("❌ PyTorch ne détecte pas CUDA")


