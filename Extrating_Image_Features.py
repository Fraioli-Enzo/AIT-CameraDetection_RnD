import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple, Optional


class ImageAnalysis:
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
        return frame
    
    @staticmethod
    def extract_colors(frame: np.ndarray, num_colors: int = 3) -> List[Tuple[int, int, int]]:
        if frame is None:
            return []
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixels = frame.reshape(-1, frame.shape[-1])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        # Check for pixels uniqueness to avoid errors if image has fewer unique colors than requested
        unique_pixels = np.unique(pixels, axis=0)
        if len(unique_pixels) < num_colors:
            num_colors = len(unique_pixels)
            if num_colors == 0:  # Handle completely blank images
                return []
        _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return [tuple(map(int, center)) for center in centers]
    
    @staticmethod
    def print_colors(colors: List[Tuple[int, int, int]]):
        for i, color in enumerate(colors):
            print(f"Color {i + 1}: {color}")

    @staticmethod
    def show_colors(colors: List[Tuple[int, int, int]], frame: Optional[np.ndarray] = None):
        # Create a display panel
        color_patch_width = 100
        panel_height = 100 * len(colors) if len(colors) > 0 else 100
        
        if frame is not None:
            # Calculate resize factor to match the height of color patches
            img_height, img_width = frame.shape[:2]
            resize_factor = panel_height / img_height
            resized_width = int(img_width * resize_factor)
            resized_img = cv2.resize(frame, (resized_width, panel_height))
            
            # Create panel with image + color patches
            panel = np.zeros((panel_height, resized_width + (color_patch_width * len(colors)), 3), dtype=np.uint8)
            panel[:, :resized_width] = resized_img
            
            # Add colors to panel
            for i, color in enumerate(colors):
                x_offset = resized_width + (i * color_patch_width)
                panel[:, x_offset:x_offset+color_patch_width] = color
                
            # Display the panel
            cv2.imshow("Image with Extracted Colors", panel)
        else:
            # Just show individual color patches if no image
            for i, color in enumerate(colors):
                color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
                color_patch[:, :] = color
                cv2.imshow(f"Color {i + 1}", color_patch)
                
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class ImagePipeline:
    @staticmethod
    def process_image(image_path: str) -> List[Tuple[int, int, int]]:
        frame = ImageAnalysis.load_image(image_path)
        colors = ImageAnalysis.extract_colors(frame, 5)
        ImageAnalysis.show_colors(colors, frame)
        ImageAnalysis.print_colors(colors)
        return colors

def main():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Select an image file")
    ImagePipeline.process_image(image_path)


if __name__ == "__main__":
    main()
 