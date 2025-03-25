from ultralytics import YOLO
import torch

def train_model():
    model = YOLO("yolov8n.pt")  # Load YOLOv8 model
    model.train(data="D:\Enzo\datasets\FabricSpotDefect\Orginal\YOLOv8\data.yaml", epochs=1, batch=16, imgsz=640)

def evaluate_model():
    model = YOLO("YOLO_V2/runs/detect/train/weights/best.pt")
    metrics = model.val()
    print(metrics)

def run_inference(image_path):
    model = YOLO("YOLO_V2/runs/detect/train/weights/best.pt")
    # La méthode predict est plus simple à utiliser
    results = model.predict(source=image_path, save=True, save_txt=True, project="YOLO_V2")
    
    # Pour afficher les résultats
    import cv2
    for r in results:
        img = r.plot()
        cv2.imshow("Detection Result", img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print(f"Results saved to output folder")

def export_model():
    model = YOLO("YOLO_V2/runs/detect/train/weights/best.pt")
    model.export(format="onnx")  # Export to ONNX
    model.export(format="engine")  # Export to TensorRT

if __name__ == "__main__":
 
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("Training on CPU. Consider using GPU for better performance.")
    
    train_model()
    print("Training completed.")
    
    evaluate_model()
    print("Evaluation completed.")
    
    export_model()
    print("Model exported.")
