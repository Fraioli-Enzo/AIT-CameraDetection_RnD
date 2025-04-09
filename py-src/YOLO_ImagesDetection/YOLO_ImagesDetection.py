from ultralytics import YOLO
import cv2
import os
import numpy as np
import tkinter as tk
import pyautogui
import ezdxf

'''
Cut100 -> overfit 
Cut50 -> medium
Cut75 -> overfit
'''

# Global variables for slider values --------------- 
brightness_value = 0        # Must be between -50 and 50
contrast_value = 1          # Must be between 0.0 and 3.0
saturation_value = 0.5      # Must be between 0.0 and 3.0
blur_value = 25             # Must be odd and between 5 and 25
threshold_value = 0.2       # Must be between 0.0 and 1.0

saturation_slider = None    # Global variable for the saturation slider
#---------------------------------------------------

# UI for camera settings ---------------------------
def create_control_panel(version):
    global saturation_slider

    def _reset_values():
        brightness_slider.set(1)
        contrast_slider.set(10)
        saturation_slider.set(10)
        blur_slider.set(5)
        threshold_slider.set(20)
    
    def _screenshot():
        # Define the folder path where the screenshot will be saved
        folder_path = "D:/Enzo/CameraDetection/py-src/YOLO_ImagesDetection/screenshots"
        
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)
        count = len([name for name in os.listdir(folder_path) if name.startswith("screenshot")])
        file_path = os.path.join(folder_path, f"screenshot_{count}.png")
        
        # Take a screenshot of the entire screen
        screenshot = pyautogui.screenshot()
        
        # Save the screenshot to the specified path
        screenshot.save(file_path)
        print(f"Screenshot saved to: {file_path}")

    # Create tkinter window
    root = tk.Tk()
    root.title(f"{version}")
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
    
    threshold_slider = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL, 
                               label="Threshold", length=300,
                               command=on_threshold_change)
    threshold_slider.set(int(threshold_value * 100))
    threshold_slider.pack(pady=5)
       
    reset_button = tk.Button(root, text="Reset to Defaults", command=_reset_values)
    reset_button.pack(pady=10)

    save_button = tk.Button(root, text="Screenshot", command=_screenshot)
    save_button.pack(pady=10)

    dynamic_red_label = tk.Label(root, text="Dynamic Text Here", font=("Helvetica", 12))
    dynamic_red_label.pack(pady=10)

    dynamic_green_label = tk.Label(root, text="Dynamic Text Here", font=("Helvetica", 12))
    dynamic_green_label.pack(pady=10)

    dynamic_blue_label = tk.Label(root, text="Dynamic Text Here", font=("Helvetica", 12))
    dynamic_blue_label.pack(pady=10)

    dynamic_brightness = tk.Label(root, text="Dynamic Text Here", font=("Helvetica", 12))
    dynamic_brightness.pack(pady=10)


    return root, dynamic_red_label, dynamic_green_label, dynamic_blue_label, dynamic_brightness

def on_threshold_change(val):
    global threshold_value
    threshold_value = float(val) / 100.0  # Normalize to range 0.0 to 1.0
    print(f"Threshold changed to: {threshold_value}")

def on_brightness_change(val):
    global brightness_value
    brightness_value = int(val) - 50  # Range -50 to 50

def on_contrast_change(val):
    global contrast_value
    contrast_value = int(val) / 10.0  # Range 0.0 to 3.0

def on_saturation_change(val):
    global saturation_value
    saturation_value = int(val) / 10.0  # Range 0.0 to 3.0

def update_saturation_slider():
    global saturation_slider, saturation_value
    if saturation_slider:
        saturation_slider.set(int(saturation_value * 10))

def on_blur_change(val):
    global blur_value
    # Ensure the value is within the range 5 to 25 and is odd
    blur_value = max(5, int(val) if int(val) % 2 == 1 else int(val) + 1)

def apply_image_adjustments(frame):
    # Convert to float and process brightness and contrast
    adjusted = frame.astype(np.float32)
    adjusted = adjusted + brightness_value
    adjusted = adjusted * contrast_value
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    # Convert to HSV for saturation adjustment
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_value
    
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply Gaussian blur
    if blur_value > 0:
        adjusted = cv2.GaussianBlur(adjusted, (blur_value, blur_value), 0)

    return adjusted

def save_binary_frame_to_dxf(binary_frame, boxes, output_path="output.dxf"):
    def is_close_to_existing(new_rect, existing_rects, threshold=20):
        for rect in existing_rects:
            # Compare each corner of the new rectangle with the existing rectangle
            for (x1, y1), (x2, y2) in zip(new_rect, rect):
                if abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold:
                    return True
        return False

    # Load or create a new DXF document
    if os.path.exists(output_path):
        doc = ezdxf.readfile(output_path)
        print(f"Loaded existing DXF file: {output_path}")
    else:
        doc = ezdxf.new()
        print(f"Created new DXF file: {output_path}")
    
    msp = doc.modelspace()

    # Extract existing rectangles from the DXF file
    existing_rectangles = set()
    for entity in msp.query("LWPOLYLINE"):
        if entity.is_closed:  # Only consider closed polylines (rectangles)
            points = tuple((int(p[0]), int(p[1])) for p in entity.get_points())
            existing_rectangles.add(points)

    # Get the width and height of the binary frame
    frame_height, frame_width = binary_frame.shape[:2]

    # Add a rectangle representing the contours of the entire image
    image_contour = ((0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height))
    if image_contour not in existing_rectangles:
        contour = msp.add_lwpolyline(image_contour, close=True)
        contour.rgb = (255, 0, 0)
        print(f"Added image contour: {image_contour}")
    else:
        print(f"Image contour already exists: {image_contour}")

    # Add new rectangles if they are not already in the DXF file and not too close to existing ones
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        y1, y2 = frame_height - y2, frame_height - y1  # Flip vertically

        new_rectangle = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        
        if new_rectangle not in existing_rectangles and not is_close_to_existing(new_rectangle, existing_rectangles):
            msp.add_lwpolyline(new_rectangle, close=True)
            existing_rectangles.add(new_rectangle)  # Add to the set of existing rectangles
            print(f"Added new rectangle: {new_rectangle}")
        else:
            print(f"Skipped rectangle (too close or already exists): {new_rectangle}")

    # Save the updated DXF file
    doc.saveas(output_path)
    print(f"DXF file updated and saved to: {output_path}")
#---------------------------------------------------

# Process camera frames and run inference ----------
def run_inference_camera(model_version_epoch):
    output_path = "D:/Enzo/CameraDetection/py-src/YOLO_ImagesDetection/dfx/detected_defects.dxf"
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing DXF file: {output_path}")
    # Get the script's directory & Build the path to the model
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, f"best{model_version_epoch}.torchscript")
    model = YOLO(model_path, task='detect')

    # Create control panel with sliders and get the root window and dynamic label
    root, dynamic_red_label, dynamic_green_label, dynamic_blue_label, dynamic_brightness = create_control_panel(model_version_epoch)

    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return
    print("Press 'q' to quit.")

    # Create named windows and set them to stay on top of other windows
    cv2.namedWindow("Camera Inference", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera Inference", cv2.WND_PROP_TOPMOST, 1)
    
    cv2.namedWindow("Histogram", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Histogram", cv2.WND_PROP_TOPMOST, 1)

    cv2.namedWindow("dfx", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("dfx", cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow("dfx", 640, 480)
    
    # Process frames in a loop
    while True:
        # Process Tkinter events to keep the UI responsive
        root.update_idletasks()
        root.update()
        
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if not ret:
            print("Error: Could not read frame from the camera.")
            break
    
        # Apply image adjustments based on slider values
        adjusted_frame = apply_image_adjustments(frame)

        # Calculte and display the histogram
        histo = live_histogram(adjusted_frame)
        cv2.imshow("Histogram", histo)

        red_mean, red_std, green_mean, green_std, blue_mean, blue_std = extract_red_channel(adjusted_frame)
        brightness = get_brightness(adjusted_frame)

        # Update the dynamic label with mean and std values
        dynamic_red_label.config(text=f"Red Channel - Mean: {red_mean:.2f}, Std: {red_std:.2f}")
        dynamic_green_label.config(text=f"Green Channel - Mean: {green_mean:.2f}, Std: {green_std:.2f}")
        dynamic_blue_label.config(text=f"Blue Channel - Mean: {blue_mean:.2f}, Std: {blue_std:.2f}")
        dynamic_brightness.config(text=f"Brightness: {brightness:.2f}")

        # Adjust saturation if brightness is below 110
        if brightness < 110:
            global saturation_value
            saturation_value = 1.2
            update_saturation_slider()
        else :
            saturation_value = 0.5
            update_saturation_slider()

        results = model.predict(source=adjusted_frame, save=False, imgsz=640, conf=threshold_value)
        for r in results:
            # Extract detections and print coordinates
            boxes = r.boxes
            if len(boxes) > 0:
                print("\n--- Detected Defects ---")
                
            for box in boxes:
                # Get coordinates (x1, y1, x2, y2 format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                
                # Put 'Class: {cls_name} |' if tere is class name in dataset with which the model have been trained
                # print(f"Confidence: {confidence:.2f} | Coordinates: ({int(x1)}, {int(y1)}); ({int(x2)}, {int(y1)}); ({int(x2)}, {int(y2)}); ({int(x1)}, {int(y2)}))") # top-left, top-right, bottom-right, bottom-left
            img = r.plot()
            # Display the mirrored image
            cv2.imshow("Camera Inference", img)


            # Create a binary frame with detected defects
            binary_frame = np.zeros_like(frame, dtype=np.uint8)

            for box in boxes:
                # Get coordinates (x1, y1, x2, y2 format)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Draw a white rectangle on the binary frame
                cv2.rectangle(binary_frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

            # Display the binary frame in the "dfx" window
            cv2.imshow("dfx", binary_frame)

            # Save the binary frame to a DXF file
            save_binary_frame_to_dxf(binary_frame, boxes, output_path)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows and destroy the Tkinter window
    cap.release()
    cv2.destroyAllWindows()
    root.destroy() 

def live_histogram(frame):
    # Calculate the histogram for each channel in the BGR color space
    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])  # Blue channel
    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])  # Green channel
    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])  # Red channel

    # Normalize the histograms to fit in the display image
    hist_b = cv2.normalize(hist_b, hist_b, 0, 255, cv2.NORM_MINMAX)
    hist_g = cv2.normalize(hist_g, hist_g, 0, 255, cv2.NORM_MINMAX)
    hist_r = cv2.normalize(hist_r, hist_r, 0, 255, cv2.NORM_MINMAX)

    # Create an image to display the histograms
    hist_image = np.zeros((300, 256, 3), dtype=np.uint8)

    # Draw the histograms
    for x in range(256):
        cv2.line(hist_image, (x, 300), (x, 300 - int(hist_b[x].item())), (255, 0, 0), 1)  # Blue
        cv2.line(hist_image, (x, 300), (x, 300 - int(hist_g[x].item())), (0, 255, 0), 1)  # Green
        cv2.line(hist_image, (x, 300), (x, 300 - int(hist_r[x].item())), (0, 0, 255), 1)  # Red

    return hist_image

def extract_red_channel(frame):
    # Extract the red channel
    red_channel = frame[:, :, 2]
    green_channel = frame[:, :, 1]
    blue_channel = frame[:, :, 0]
    
    # Calculate mean and standard deviation for each channel
    red_mean = np.mean(red_channel)
    red_std = np.std(red_channel)

    green_mean = np.mean(green_channel)
    green_std = np.std(green_channel)

    blue_mean = np.mean(blue_channel)
    blue_std = np.std(blue_channel)

    return red_mean, red_std, green_mean, green_std, blue_mean, blue_std

def get_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()  # V channel
    return brightness
#---------------------------------------------------


if __name__ == "__main__":
    # Create a dictionary mapping indices to model names
    models = {
        "1": "Small25_v8",
        "2": "Small50_v8",
        "3": "Small25_v11",
        "4": "Small50_v11",
        "5": "Cut50",
        "6": "newSmall25_v8",
        "7": "newSmall25_v11", ## Good one
        "8": "newSmall50_v11",
        "9": "newSmall25_v10", ## Also Good one
    }
    
    print("Choose the model you want to use:")
    for idx, model in models.items():
        print(f"{idx}. {model}")
    
    # Get user input with default value
    model_index = input("Enter the model index (default is 7): ").strip()
    
    # Use default if input is empty or invalid
    if not model_index or model_index not in models:
        model_index = "7"
        print("Using default model: newSmall25_v11")
    
    # Set the model_version_epoch based on the selected index
    model_version_epoch = models[model_index]
    print(f"Selected model: {model_version_epoch}")
    run_inference_camera(model_version_epoch)