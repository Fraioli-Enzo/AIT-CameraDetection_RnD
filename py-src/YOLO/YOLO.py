from ultralytics import YOLO
import os

def train_model():
    model = YOLO("yolov8n.pt")  # Load YOLOv8 model
    model.train(data="D:\Enzo\datasets\FabricSpotDefect\Augmented\YOLOv8\data.yaml", epochs=50, batch=16, imgsz=640)

def evaluate_model():
    model = YOLO("py-src/YOLO/runs/detect/train/weights/best.pt")
    metrics = model.val()
    print(metrics)

def run_inference(image_path):
    model = YOLO("py-src/YOLO/runs/detect/train/weights/best.pt")
    # La méthode predict est plus simple à utiliser
    results = model.predict(source=image_path, save=True, save_txt=True, project="py-src/YOLO/output")
    
    # Pour afficher les résultats
    import cv2
    for r in results:
        img = r.plot()
        cv2.imshow("Detection Result", img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print(f"Results saved to output folder")

def export_model():
    model = YOLO("py-src/YOLO/runs/detect/train/weights/best.pt")
    model.export(format="onnx")  # Export to ONNX
    model.export(format="engine")  # Export to TensorRT

if __name__ == "__main__":
 
    # if torch.cuda.is_available():
    #     print("CUDA is available. Training on GPU.")
    # else:
    #     print("Training on CPU. Consider using GPU for better performance.")
    
    # train_model()
    # print("Training completed.")
    
    # evaluate_model()
    # print("Evaluation completed.")
    
    test_image = "Images/anomali_15.png"  # Replace with an actual test image
    if os.path.exists(test_image):
        run_inference(test_image)
        print("Inference completed.")
    else:
        print(f"Test image {test_image} not found.")
    
    # export_model()
    # print("Model exported.")
