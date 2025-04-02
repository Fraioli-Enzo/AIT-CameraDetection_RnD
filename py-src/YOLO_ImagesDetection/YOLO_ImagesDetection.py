from ultralytics import YOLO
import cv2
import os
import numpy as np
import time 
import tkinter as tk


'''
Cut100 -> overfit 
Cut50 -> medium
Cut75 -> overfit

THIS IS A TEST
'''

# Global variables for slider values
brightness_value = 0
contrast_value = 1
saturation_value = 1
blur_value = 5
threshold_value = 0.2

def create_control_panel(version):
    # Create tkinter window
    root = tk.Tk()
    root.title(f"Camera Controls - {version}")
    root.attributes('-topmost', True)  # Make window stay on top
    
    # Set the initial slider values
    brightness_slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, 
                                label="Brightness", length=300,
                                command=on_brightness_change)
    brightness_slider.set(1) 
    brightness_slider.pack(pady=5)
    
    contrast_slider = tk.Scale(root, from_=0, to=30, orient=tk.HORIZONTAL, 
                              label="Contrast", length=300,
                              command=on_contrast_change)
    contrast_slider.set(int(contrast_value * 10))  # Convert from 2.0 to 20
    contrast_slider.pack(pady=5)
    
    saturation_slider = tk.Scale(root, from_=0, to=30, orient=tk.HORIZONTAL, 
                                label="Saturation", length=300,
                                command=on_saturation_change)
    saturation_slider.set(int(saturation_value * 10))  # Convert from 2.0 to 20
    saturation_slider.pack(pady=5)
    
    blur_slider = tk.Scale(root, from_=0, to=25, orient=tk.HORIZONTAL, 
                          label="Blur", length=300,
                          command=on_blur_change)
    blur_slider.set(blur_value)
    blur_slider.pack(pady=5)
    
    threshold_slider = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, 
                               label="Threshold", length=300,
                               command=on_threshold_change)
    threshold_slider.set(int(threshold_value * 100))
    threshold_slider.pack(pady=5)
    
    return root

# Callback function for threshold slider
def on_threshold_change(val):
    global threshold_value
    threshold_value = val / 100.0  # Normalize to range 0.0 to 1.0

# Callback functions for sliders
def on_brightness_change(val):
    global brightness_value
    brightness_value = int(val) - 50  # Range -50 to 50

def on_contrast_change(val):
    global contrast_value
    contrast_value = int(val) / 10.0  # Range 0.0 to 3.0

def on_saturation_change(val):
    global saturation_value
    saturation_value = int(val) / 10.0  # Range 0.0 to 3.0

def on_blur_change(val):
    global blur_value
    # Ensure the value is within the range 5 to 25 and is odd
    blur_value = max(5, int(val) if int(val) % 2 == 1 else int(val) + 1)

def apply_image_adjustments(frame):
    # Convert to float for processing
    adjusted = frame.astype(np.float32)
    
    # Apply brightness
    adjusted = adjusted + brightness_value
    
    # Apply contrast
    adjusted = adjusted * contrast_value
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    # Convert to HSV for saturation adjustment
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply saturation
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_value
    
    
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply Gaussian blur
    if blur_value > 0:
        adjusted = cv2.GaussianBlur(adjusted, (blur_value, blur_value), 0)
    
    return adjusted

def run_inference_camera(model_version_epoch):
    # Get the script's directory
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Build the path to the model
    model_path = os.path.join(base_path, f"best{model_version_epoch}.torchscript")

    # Load the YOLO model
    model = YOLO(model_path, task='detect')

    # Create control panel with sliders and get the root window
    root = create_control_panel(model_version_epoch)

    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'q' to quit.")

    # Create named windows and set them to stay on top
    cv2.namedWindow("Camera Inference", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera Inference", cv2.WND_PROP_TOPMOST, 1)
    
    cv2.namedWindow("Adjusted Frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Adjusted Frame", cv2.WND_PROP_TOPMOST, 1)
    
    # FPS limiting variables
    target_fps = 24
    frame_time = 1.0 / target_fps
    
    while True:
        # Process Tkinter events to keep the UI responsive
        root.update_idletasks()
        root.update()
        
        # Track frame start time
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break
        
        # Apply image adjustments based on slider values
        adjusted_frame = apply_image_adjustments(frame)
        
        # Run inference on the adjusted frame
        results = model.predict(source=adjusted_frame, save=False, imgsz=640, conf=threshold_value)

        # Display the results
        for r in results:
            # Extract detections and print coordinates
            boxes = r.boxes
            if len(boxes) > 0:
                print("\n--- Detected Objects ---")
                
            for box in boxes:
                # Get coordinates (x1, y1, x2, y2 format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence score
                confidence = box.conf[0].item()
                
                # Get class name
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                
                # Print information
                print(f"Class: {cls_name} | Confidence: {confidence:.2f} | Coordinates: ({int(x1)}, {int(y1)}); ({int(x2)}, {int(y1)}); ({int(x2)}, {int(y2)}); ({int(x1)}, {int(y2)});)") # top-left, top-right, bottom-right, bottom-left
            
            img = r.plot()
            cv2.imshow("Camera Inference", img)
        
        # Also display the original frame with adjustments for comparison
        cv2.imshow("Adjusted Frame", adjusted_frame)

        # Calculate elapsed time for this frame
        elapsed = time.time() - frame_start
        
        # If we processed the frame too quickly, wait to maintain target FPS
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
            
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()  # Properly destroy the Tkinter window

if __name__ == "__main__":
    # Create a dictionary mapping indices to model names
    models = {
        "1": "Small25_v8",
        "2": "Small25_v11",
        "3": "Small50_v8",
        "4": "Small50_v11"
    }
    
    print("Choose the model you want to use:")
    for idx, model in models.items():
        print(f"{idx}. {model}")
    
    # Get user input with default value
    model_index = input("Enter the model index (default is 1): ").strip()
    
    # Use default if input is empty or invalid
    if not model_index or model_index not in models:
        model_index = "1"
        print("Using default model: Small25_v8")
    
    # Set the model_version_epoch based on the selected index
    model_version_epoch = models[model_index]
    print(f"Selected model: {model_version_epoch}")
    run_inference_camera(model_version_epoch)