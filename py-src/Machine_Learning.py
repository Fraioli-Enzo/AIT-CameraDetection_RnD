import cv2
import tkinter as tk
from tkinter import filedialog
from torchvision.models import alexnet
from torch import nn
import numpy as np
import torch

class DataProcessing:
    @staticmethod
    def image_loader(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        return image
    
    @staticmethod
    def noise_removal(image, patch_size=5, h=5):
        """
        Perform basic noise removal using OpenCV's implementation of Non-Local Means Denoising.

        Parameters:
        - image: Input noisy image (numpy array).
        - patch_size: Size of the patches to compare (default is 7).
        - filter_sigma: Standard deviation for Gaussian filter (default is 3).
        - h: Filtering parameter, also known as h parameter in OpenCV's fastNlMeansDenoising (default is 10).

        Returns:
        - Denoised image (numpy array).
        """
        # Check if image is grayscale or color
        if len(image.shape) == 2:
            # For grayscale images
            denoised_image = cv2.fastNlMeansDenoising(
                image,
                None,
                h=h,
                templateWindowSize=patch_size,
                searchWindowSize=patch_size*3
            )
        else:
            # For color images
            denoised_image = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=h,
                hColor=h,
                templateWindowSize=patch_size,
                searchWindowSize=patch_size*3
            )
            
        return denoised_image

    @staticmethod
    def alignement(image, template):
        """
        TODO
        Perform template matching to align the input image with a template.

        Parameters:
        - image: Input image (numpy array).
        - template: Template image (numpy array).

        Returns:
        - Aligned image (numpy array).
        """     

    @staticmethod
    def visualize_image(image):
        """
        Display the input image.

        Parameters:
        - image: Input image (numpy array).
        """     
        print("Press any key to close the image window.")
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class FeatureExtraction:
    @staticmethod
    def CNN_model_selection():
        """
        Select a pre-trained CNN model for feature extraction.

        Returns:
        - Pre-trained CNN model.
        """
        model = alexnet()
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model
    
    @staticmethod
    def features_extraction(model, image):
        """
        Extract features from the input image using the pre-trained CNN model.

        Parameters:
        - model: Pre-trained CNN model.
        - image: Input image (numpy array).

        Returns:
        - Extracted features (numpy array).
        """
        image = cv2.resize(image, (224, 224))  # Resize to match AlexNet input size
        image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W) format
        image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model(image)  # Extract features
        return features
    



class Pipeline:
    @staticmethod
    def process_pattern(image_path):  
        image = DataProcessing.image_loader(image_path)
        denoised_image = DataProcessing.noise_removal(image)
        DataProcessing.visualize_image(denoised_image)
        model = FeatureExtraction.CNN_model_selection()
        features = FeatureExtraction.features_extraction(model, denoised_image)
        print(features)



def main():
    # Load the image
    tk.Tk().withdraw()
    image_path = filedialog.askopenfilename()
    Pipeline.process_pattern(image_path)

    
if __name__ == "__main__":
    main()
