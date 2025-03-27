from ultralytics import YOLO
import cv2
import os

def run_inference_camera(model_version_epoch="25"):
    # Get the script's directory
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Build the path to the model
    model_path = os.path.join(base_path, f"best{model_version_epoch}.torchscript")
    results_path = os.path.join(base_path, "predicts")

    # Load the YOLO model
    model = YOLO(model_path, task='detect')

    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        # Run inference on the current frame
        results = model.predict(source=frame, save=False, imgsz=640)

        # Display the results
        for r in results:
            img = r.plot()
            cv2.imshow("Camera Inference", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_version_epoch = input("Enter the model version epoch (default is 25): ")
    run_inference_camera(model_version_epoch)