import os
import torch
import tkinter as tk
from PIL import Image
from tkinter import filedialog
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import AlexNet_Weights

class ImageClassifier:
    def __init__(self, model_name='alexnet'):
        """
        Initialize the image classifier with the specified model.
        
        Args:
            model_name (str): Name of the model to use, default is 'alexnet'
        """
        # Load the pretrained AlexNet model
        self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        
        # Put the model in evaluation mode
        self.model.eval()
        
        # Check if CUDA is available and move model to GPU if it is
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Load the class labels
        with open('py-src/AlexNet_1k_Classes/imagenet_classes.txt') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
    def load_image(self, image_path):
        """
        Load and preprocess an image for classification.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image_path):
        """
        Classify an image and return the top 5 predictions.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            list: List of tuples containing (class_name, probability)
        """
        # Load and preprocess the image
        img_tensor = self.load_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Convert to list of (class_name, probability) tuples
        # Only include classes that are in our list
        results = []
        for idx, prob in zip(top5_idx, top5_prob):
            idx_value = idx.item()
            if idx_value < len(self.classes):
                results.append((self.classes[idx_value], prob.item()))
            else:
                results.append((f"Unknown Class {idx_value}", prob.item()))
        
        return results
    
    def batch_predict(self, directory):
        """
        Classify all images in a directory.
        
        Args:
            directory (str): Path to directory containing images
        
        Returns:
            dict: Dictionary mapping image filenames to prediction results
        """
        results = {}
        valid_extensions = ['.jpg', '.jpeg', '.png']
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Skip if not a file or not an image
            if not os.path.isfile(file_path):
                continue
                
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue
                
            try:
                predictions = self.predict(file_path)
                results[filename] = predictions
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
        return results

def main():
    tk.Tk().withdraw()  # Hide the main window
    # Example usage
    classifier = ImageClassifier()
    
    # Single image classification
    image_path = filedialog.askopenfilename()
    if os.path.exists(image_path):
        results = classifier.predict(image_path)
        print(f"Results for {image_path}:")
        for class_name, probability in results:
            print(f"{class_name}: {probability:.4f}")
    
    # # Batch classification
    # image_dir = filedialog.askdirectory()
    # if os.path.isdir(image_dir):
    #     batch_results = classifier.batch_predict(image_dir)
    #     print(f"\nBatch results for directory {image_dir}:")
    #     for filename, predictions in batch_results.items():
    #         print(f"\n{filename}:")
    #         for class_name, probability in predictions:
    #             print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()