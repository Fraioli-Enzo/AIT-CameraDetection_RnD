from ultralytics import YOLO
import cv2
import os
import numpy as np
import time  # Add time module for FPS limiting

'''
Cut100 -> overfit 
Cut50 -> medium
Cut75 -> overfit
'''

# Global variables for slider values
brightness_value = 0
contrast_value = 2
saturation_value = 2
blur_value = 5
threshold_value = 0.2

def create_control_panel(version):
    # Create a window for the sliders
    cv2.namedWindow(version)
    
    # Create sliders
    cv2.createTrackbar('Brightness', version, 0, 100, on_brightness_change)
    cv2.createTrackbar('Contrast', version, 10, 30, on_contrast_change)
    cv2.createTrackbar('Saturation', version, 10, 30, on_saturation_change)
    cv2.createTrackbar('Blur', version, 5, 25, on_blur_change)
    cv2.createTrackbar('Threshold', version, int(threshold_value * 100), 100, on_threshold_change) 

# Callback function for threshold slider
def on_threshold_change(val):
    global threshold_value
    threshold_value = val / 100.0  # Normalize to range 0.0 to 1.0

# Callback functions for sliders
def on_brightness_change(val):
    global brightness_value
    brightness_value = val - 50  # Range -50 to 50

def on_contrast_change(val):
    global contrast_value
    contrast_value = val / 10.0  # Range 0.0 to 3.0

def on_saturation_change(val):
    global saturation_value
    saturation_value = val / 10.0  # Range 0.0 to 3.0

def on_blur_change(val):
    global blur_value
    # Ensure the value is within the range 5 to 25 and is odd
    blur_value = max(5, val if val % 2 == 1 else val + 1)

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

def run_inference_camera(model_version_epoch="25"):
    # Get the script's directory
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Build the path to the model
    model_path = os.path.join(base_path, f"best{model_version_epoch}.torchscript")
    results_path = os.path.join(base_path, "predicts")

    # Load the YOLO model
    model = YOLO(model_path, task='detect')

    # Create control panel with sliders
    create_control_panel(version=model_version_epoch)

    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'q' to quit.")
    
    # FPS limiting variables
    target_fps = 24
    frame_time = 1.0 / target_fps
    
    while True:
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

if __name__ == "__main__":
    model_version_epoch = input("Enter the model version epoch (default is 25): ")
    run_inference_camera(model_version_epoch)