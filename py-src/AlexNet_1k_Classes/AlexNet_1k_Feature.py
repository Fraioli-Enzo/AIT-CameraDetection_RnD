import os
import torch
import numpy as np
import tkinter as tk
from PIL import Image
from tkinter import filedialog
import matplotlib.pyplot as plt
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
    
class AlexNetFeatureExtractor:
    def __init__(self, classifier):
        """
        Initialize the feature extractor with a reference to the classifier.
        
        Args:
            classifier (ImageClassifier): An instance of the ImageClassifier class
        """
        self.classifier = classifier
    
    def visualize_first_layer_features(self, image_path, output_dir="py-src/AlexNet_1k_Classes/feature_maps", threshold_percentile=80):
        """
        Visualize the feature maps from all convolutional layers for a given image.
        
        Args:
            image_path (str): Path to the image file
            output_dir (str): Directory to save the feature maps
        
        Returns:
            list: Paths to the saved feature maps for each layer
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess the image
        img_tensor = self.classifier.load_image(image_path)
        
        # Indices of convolutional layers in AlexNet
        conv_indices = [0, 3]
        
        output_paths = []

        output_dir_features = os.path.join(output_dir, os.path.basename(image_path).split('.')[0])
        os.makedirs(output_dir_features, exist_ok=True)
        
        for layer_idx in conv_indices:
            # Create a model that outputs the activations up to this conv layer
            feature_extractor = torch.nn.Sequential(*list(self.classifier.model.features[:layer_idx+1]))
            
            # Extract features
            with torch.no_grad():
                features = feature_extractor(img_tensor)
            
            # Convert to numpy for visualization
            feature_maps = features.squeeze(0).cpu().numpy()
            
            # Plot the original image
            img = Image.open(image_path)
            plt.figure(figsize=(12, 12))
            plt.subplot(8, 8, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.axis('off')
            
            # Plot the feature maps (show up to 63 filters)
            num_maps = min(63, feature_maps.shape[0])
            for i in range(num_maps):
                plt.subplot(8, 8, i+2)
                
                # Get current feature map
                feature_map = feature_maps[i]
                # Calculate threshold based on percentile
                threshold = np.percentile(feature_map, threshold_percentile)
                # Create a masked version showing only bright spots
                masked_map = np.copy(feature_map)
                masked_map[masked_map < threshold] = np.nan  # Set below-threshold to NaN (will be transparent)
                # Create a custom colormap with transparency for low values
                cmap = plt.cm.gray.copy()
                cmap.set_bad('white', alpha=0)  # NaN values become transparent
                
                # Plot with transparency for low values
                plt.imshow(masked_map, cmap=cmap)
                plt.title(f"Filter {i+1}")
                plt.axis('off')
            
            # Save the figure
            layer_name = f"layer_{layer_idx}"
            output_path = os.path.join(output_dir_features, f"{os.path.basename(image_path).split('.')[0]}_{layer_name}_features.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            print(f"Feature maps for layer {layer_idx} saved to {output_path}")
            output_paths.append(output_path)
        
        return output_paths

    def combine_image(self, image_path1, image_path2, output_dir="py-src/AlexNet_1k_Classes/combine_feature_maps"):
        Image1 = Image.open(image_path1)
        Image2 = Image.open(image_path2)
        # Ensure both images are in RGB mode
        Image1 = Image1.convert('RGB')
        Image2 = Image2.convert('RGB')

        # Convert the images from black to red tint
        def convert_to_red(img):
            # Convert to numpy array
            img_array = np.array(img)
            # Keep only the red channel, zero out green and blue
            red_img = img_array.copy()
            red_img[:, :, 1] = 0  # Zero out green channel
            red_img[:, :, 2] = 0  # Zero out blue channel
            return Image.fromarray(red_img)

        Image2 = convert_to_red(Image2)

        # Make sure both images are the same size
        width = max(Image1.width, Image2.width)
        height = max(Image1.height, Image2.height)
        Image1 = Image1.resize((width, height))
        Image2 = Image2.resize((width, height))

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Combine the images
        supperposition = Image.blend(Image1, Image2, alpha=0.5)
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path1).split('.')[0]}_{os.path.basename(image_path2).split('.')[0]}_combined.png")
        supperposition.save(output_path)
        print(f"Combined image saved to {output_path}")


def main():
    tk.Tk().withdraw()  # Hide the main window

    print("\033[31mInitializing AlexNet model please wait.\033[0m")
    classifier = ImageClassifier()
    feature_extractor = AlexNetFeatureExtractor(classifier)
    print("\033[32mAlexNet model initialized.\033[0m")  # Print in green color

    print("\nOptions:")
    print("1: Single image classification")
    print("2: Folder classification")
    print("3: Visualize first conv layer features")
    print("4: Combine two images for comparaison")

    mode = int(input("Enter your choice (1-4): "))
    if mode == 1:
        # Single image classification
        image_path = filedialog.askopenfilename()
        if os.path.exists(image_path):
            results = classifier.predict(image_path)
            print(f"Results for {image_path}:")
            for class_name, probability in results:
                print(f"{class_name}: {probability:.4f}")

    elif mode == 2:
        # Batch classification
        image_dir = filedialog.askdirectory()
        if os.path.isdir(image_dir):
            batch_results = classifier.batch_predict(image_dir)
            print(f"\nBatch results for directory {image_dir}:")
            for filename, predictions in batch_results.items():
                print(f"\n{filename}:")
                for class_name, probability in predictions:
                    print(f"{class_name}: {probability:.4f}")

    elif mode == 3:
        # Feature visualization
        image_path = filedialog.askopenfilename()
        if os.path.exists(image_path):
            output_path = feature_extractor.visualize_first_layer_features(image_path, "py-src/AlexNet_1k_Classes/feature_maps")
            print(f"Visualization complete! Check the output at: features_maps")

    elif mode == 4:
        # Feature visualization
        image_path1 = filedialog.askopenfilename()
        image_path2 = filedialog.askopenfilename()
        if os.path.exists(image_path1) and os.path.exists(image_path2):
            feature_extractor.combine_image(image_path1, image_path2, "py-src/AlexNet_1k_Classes/combine_feature_maps")


if __name__ == "__main__":
    main()