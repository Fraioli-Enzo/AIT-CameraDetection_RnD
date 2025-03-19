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

# Option 1 & 2
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
    
# Option 3
class AlexNetFeatureExtractor:
    def __init__(self, classifier):
        """
        Initialize the feature extractor with a reference to the classifier.
        
        Args:
            classifier (ImageClassifier): An instance of the ImageClassifier class
        """
        self.classifier = classifier
    
    def visualize_layer_features(self, image_path, output_dir="py-src/AlexNet_1k_Classes/feature_maps", threshold_percentile=80):
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

    ## Not use, keep just in case ##
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

# Option 4
class PeriodicityAnalyzer:
    def __init__(self, classifier):
        """
        Initialize the periodicity analyzer with a reference to the classifier.
        
        Args:
            classifier (ImageClassifier): An instance of the ImageClassifier class
        """
        self.classifier = classifier

    def visualize_periodicity_map(self, image_path, layer_idx=0, output_dir="py-src/AlexNet_1k_Classes/periodicity_maps"):
        """
        Create a visualization that highlights periodic patterns in the original image.
        
        Args:
            image_path (str): Path to the image
            layer_idx (int): Layer index to use for analysis
            output_dir (str): Directory to save the visualization
            
        Returns:
            str: Path to the saved visualization
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and process the image
        img_tensor = self.classifier.load_image(image_path)
        original_img = Image.open(image_path)
        img_np = np.array(original_img)
        
        # Extract features for the specified layer
        feature_extractor = torch.nn.Sequential(*list(self.classifier.model.features[:layer_idx+1]))
        with torch.no_grad():
            features = feature_extractor(img_tensor)
        feature_maps = features.squeeze(0).cpu().numpy()
        
        # Create a composite periodicity map
        periodicity_map = np.zeros((feature_maps.shape[1], feature_maps.shape[2]))
        peak_strength_map = np.zeros_like(periodicity_map)
        
        # For each feature map
        for i in range(feature_maps.shape[0]):
            feature_map = feature_maps[i]
            
            # Apply FFT
            fft_result = np.fft.fft2(feature_map)
            fft_shifted = np.fft.fftshift(fft_result)
            magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
            
            # Find peaks in frequency domain
            from scipy.signal import find_peaks
            flat_spectrum = magnitude_spectrum.flatten()
            threshold = np.mean(flat_spectrum) + 2*np.std(flat_spectrum)
            peaks, properties = find_peaks(flat_spectrum, height=threshold)
            
            if len(peaks) > 0:
                # Reconstruct just the periodicity using inverse FFT
                # Zero out non-peak frequencies
                filtered_spectrum = np.zeros_like(fft_shifted, dtype=complex)
                for p in peaks:
                    # Convert 1D peak index back to 2D coordinates
                    y, x = np.unravel_index(p, fft_shifted.shape)
                    # Copy a small region around the peak
                    window_size = 3
                    y_min, y_max = max(0, y-window_size), min(fft_shifted.shape[0], y+window_size+1)
                    x_min, x_max = max(0, x-window_size), min(fft_shifted.shape[1], x+window_size+1)
                    filtered_spectrum[y_min:y_max, x_min:x_max] = fft_shifted[y_min:y_max, x_min:x_max]
                
                # Inverse FFT to get periodic component
                inverse_fft = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))
                periodic_component = np.abs(inverse_fft)
                
                # Normalize
                if periodic_component.max() > 0:
                    periodic_component = periodic_component / periodic_component.max()
                
                # Add to composite maps
                peak_strength = np.sum(properties['peak_heights'])
                periodicity_map += periodic_component * peak_strength
                peak_strength_map += peak_strength
        
        # Normalize the composite map
        if peak_strength_map.max() > 0:
            periodicity_map = periodicity_map / peak_strength_map.max()
        
        # Resize to match original image
        from scipy.ndimage import zoom
        zoom_factors = (img_np.shape[0] / periodicity_map.shape[0], 
                    img_np.shape[1] / periodicity_map.shape[1])
        resized_map = zoom(periodicity_map, zoom_factors, order=1)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis('off')
        
        # Periodicity heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(resized_map, cmap='hot')
        plt.title("Periodicity Map")
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(original_img)
        plt.imshow(resized_map, cmap='hot', alpha=0.6)
        plt.title("Overlay")
        plt.axis('off')
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}-perio_maps.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path


def main():
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()  # Hide the root window

    print("\033[31mInitializing AlexNet model please wait.\033[0m")
    classifier = ImageClassifier()
    feature_extractor = AlexNetFeatureExtractor(classifier)
    periodicity_info = PeriodicityAnalyzer(classifier)
    print("\033[32mAlexNet model initialized.\033[0m")

    print("\033[93m\nOptions:\033[0m")
    print("1: Image classification (1000 classes) - Base pretrained AlexNet")
    print("2: Visualize first conv layer features map")
    print("3: Analyze image periodicity")
    
    mode = int(input("Enter your choice (1-3): "))
    if mode == 1:
        print("\033[93m\nOptions:\033[0m")
        print("1: Single image classification - Base pretrained AlexNet")
        print("2: Folder classification - Base pretrained AlexNet")

        option = int(input("Enter your choice 1 or 2: "))
        if option == 1:
            # Single image classification
            image_path = filedialog.askopenfilename(title="Select image to analyze")
            if os.path.exists(image_path):
                results = classifier.predict(image_path)
                print(f"\nResults for {image_path}:")
                for class_name, probability in results:
                    print(f"{class_name}: {probability:.4f}")

        elif option == 2:
            # Batch classification
            image_dir = filedialog.askdirectory(title="Select folder to analyze")
            if os.path.isdir(image_dir):
                batch_results = classifier.batch_predict(image_dir)
                print(f"\nBatch results for directory {image_dir}:")
                for filename, predictions in batch_results.items():
                    print(f"\n{filename}:")
                    for class_name, probability in predictions:
                        print(f"{class_name}: {probability:.4f}")

    elif mode == 2:
        # Feature visualization
        image_path = filedialog.askopenfilename(title="Select image to analyze")
        if os.path.exists(image_path):
            output_path = feature_extractor.visualize_layer_features(image_path, "py-src/AlexNet_1k_Classes/feature_maps")
            print(f"\033[32m\nVisualization complete! Check the output at {output_path}\033[0m (crtl + click to open)")

    elif mode == 3:
        # Analyze single image periodicity
        image_path = filedialog.askopenfilename(title="Select image to analyze")
        if os.path.exists(image_path):
            print("\nCreating periodicity map...")
            map_path = periodicity_info.visualize_periodicity_map(image_path)  
            print(f"\033[32mPeriodicity map saved to {map_path}\033[0m (crtl + click to open)")

if __name__ == "__main__":
    main()