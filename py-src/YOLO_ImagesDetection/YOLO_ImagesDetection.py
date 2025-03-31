from ultralytics import YOLO
import cv2
import os
import numpy as np

'''
Cut100 -> overfit 
Cut50 -> medium
Cut75 -> overfit
'''

# Global variables for slider values
brightness_value = 0
contrast_value = 1
saturation_value = 1
blur_value = 5

def create_control_panel():
    # Create a window for the sliders
    cv2.namedWindow('Camera Controls')
    
    # Create sliders
    cv2.createTrackbar('Brightness', 'Camera Controls', 0, 100, on_brightness_change)
    cv2.createTrackbar('Contrast', 'Camera Controls', 10, 30, on_contrast_change)
    cv2.createTrackbar('Saturation', 'Camera Controls', 10, 30, on_saturation_change)
    cv2.createTrackbar('Blur', 'Camera Controls', 5, 25, on_blur_change)

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
    blur_value = val if val % 2 == 1 else val + 1  # Ensure odd number for Gaussian kernel


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
    create_control_panel()

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
        
        # Apply image adjustments based on slider values
        adjusted_frame = apply_image_adjustments(frame)
        
        # Run inference on the adjusted frame
        results = model.predict(source=adjusted_frame, save=False, imgsz=640)

        # Display the results
        for r in results:
            img = r.plot()
            cv2.imshow("Camera Inference", img)
        
        # Also display the original frame with adjustments for comparison
        cv2.imshow("Adjusted Frame", adjusted_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_version_epoch = input("Enter the model version epoch (default is 25): ")
    run_inference_camera(model_version_epoch)