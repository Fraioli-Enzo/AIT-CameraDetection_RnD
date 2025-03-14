import torch
import torchvision.models as models
from torchvision.models import AlexNet_Weights
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import torch.nn as nn

class SingleLayerVisualizer:
    def __init__(self, target_layer='conv2', layer_index=3):
        """
        Initialize the feature map visualizer with the AlexNet model for single layer analysis
        
        Args:
            target_layer (str): Name of the target layer for visualization
            layer_index (int): Index of the layer in AlexNet features
        """
        self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.eval()
        
        # Extract convolutional layers only
        self.features = self.model.features
        self.layer_name = target_layer
        self.layer_index = layer_index
        self.activation = {}
        
        # Register hook
        self.features[layer_index].register_forward_hook(self._get_activation_hook())
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def _get_activation_hook(self):
        """Create a hook to capture layer activations"""
        def hook(model, input, output):
            self.activation[self.layer_name] = output.detach()
        return hook
    
    def load_image(self, image_path):
        """
        Load and preprocess an image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert("RGB")
        self.input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return self.input_tensor
    
    def extract_features(self):
        """Extract features using the model"""
        with torch.no_grad():
            self.features(self.input_tensor)
        
        # Extract the activation map from the target layer
        self.activation_map = self.activation[self.layer_name].squeeze(0)
        return self.activation_map
    
    def find_top_activations(self, k=100):
        """
        Find the top k activation points
        
        Args:
            k (int): Number of top activations to find
            
        Returns:
            tuple: (top values, coordinates in activation map, coordinates in original image)
        """
        # Get maximum activation across all channels
        max_channel = self.activation_map.max(dim=0)[0]
        
        # Get top k activations
        flattened = max_channel.flatten()
        top_values, top_indices = torch.topk(flattened, k=k)
        
        # Convert flat indices to 2D coordinates
        h, w = max_channel.shape
        top_coords = [(idx // w, idx % w) for idx in top_indices.tolist()]
        
        # Scale coordinates to original image size
        orig_width, orig_height = 224, 224
        activation_map_size = self.activation_map.shape[1:]  # (Height, Width)
        
        top_coords_scaled = [
            (int((y / activation_map_size[0]) * orig_height),
             int((x / activation_map_size[1]) * orig_width))
            for (y, x) in top_coords
        ]
        
        return top_values, top_coords, top_coords_scaled
    
    def show_original_image(self):
        """Display the original image"""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.input_tensor.squeeze().permute(1, 2, 0).numpy())
        plt.title("Original Image")
        plt.show()
    
    def visualize_activation_map(self, top_coords=None):
        """
        Visualize the activation map
        
        Args:
            top_coords (list, optional): List of coordinates to highlight
        """
        max_channel = self.activation_map.max(dim=0)[0]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(max_channel.numpy(), cmap='jet')
        
        if top_coords:
            for coord in top_coords:
                plt.scatter(coord[1], coord[0], color='red', marker='x')
                
        plt.colorbar()
        plt.title(f"Activation Map from {self.layer_name} Layer")
        plt.show()
    
    def visualize_image_with_activations(self, top_coords_scaled):
        """
        Visualize the original image with activation points
        
        Args:
            top_coords_scaled (list): List of coordinates in original image scale
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.input_tensor.squeeze().permute(1, 2, 0).numpy())
        
        for y, x in top_coords_scaled:
            plt.scatter(x, y, color='red', marker='x', s=100)
            
        plt.title("Original Image with Top Activation Points")
        plt.show()
    
    def analyze_image(self, image_path, k=100):
        """
        Complete analysis pipeline
        
        Args:
            image_path (str): Path to the image
            k (int): Number of top activations to find
        """
        # Load and show original image
        self.load_image(image_path)
        self.show_original_image()
        
        # Extract features
        self.extract_features()
        
        # Find top activations
        top_values, top_coords, top_coords_scaled = self.find_top_activations(k)
        
        # Print activation information
        print(f"Top {k} Activation Values: {top_values.tolist()}")
        print(f"Activation map size: {self.activation_map.shape[1:]}")
        print(f"Top {k} Activation Coordinates in Original Image: {top_coords_scaled}")
        
        # Visualize results
        self.visualize_activation_map(top_coords)
        self.visualize_image_with_activations(top_coords_scaled)

class SPPLayer(nn.Module):
    def __init__(self, levels):
        super(SPPLayer, self).__init__()
        self.levels = levels

    def forward(self, x):
        num_samples, num_channels, h, w = x.size()
        pyramid = []
        for level in self.levels:
            kernel_size = (h // level, w // level)
            stride = kernel_size
            pooling = nn.AdaptiveMaxPool2d(kernel_size)
            pyramid.append(pooling(x).view(num_samples, -1))
        return torch.cat(pyramid, dim=1)
    
class SingleLayerVisualizer:
    def __init__(self, target_layer='conv2', layer_index=3, spp_levels=[1, 2, 4]):
        """
        Initialize the feature map visualizer with the AlexNet model for single layer analysis
        
        Args:
            target_layer (str): Name of the target layer for visualization
            layer_index (int): Index of the layer in AlexNet features
            spp_levels (list): Levels for the SPP layer
        """
        self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.eval()
        
        # Extract convolutional layers only
        self.features = self.model.features
        self.layer_name = target_layer
        self.layer_index = layer_index
        self.activation = {}
        
        # Register hook
        self.features[layer_index].register_forward_hook(self._get_activation_hook())
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # SPP layer
        self.spp = SPPLayer(spp_levels)
    
    def extract_features(self):
        """Extract features using the model"""
        with torch.no_grad():
            self.features(self.input_tensor)
        
        # Extract the activation map from the target layer
        self.activation_map = self.activation[self.layer_name].squeeze(0)
        
        # Apply SPP layer
        self.spp_features = self.spp(self.activation_map.unsqueeze(0))
        return self.activation_map
    
class MultiLayerVisualizer:
    def __init__(self, spp_levels=[1, 2, 4]):
        """Initialize the feature map visualizer with the AlexNet model for multi-layer analysis"""
        self.model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.model.eval()
        
        # Extract convolutional layers only, don't take classification layers
        self.features = self.model.features
        
        # Define all conv layers in AlexNet
        self.conv_layers = {
            'conv1': 0,
            'conv2': 3,
            'conv3': 8,
            'conv4': 10,
            'conv5': 12
        }
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.activation = {}
        
        # SPP layer
        self.spp = SPPLayer(spp_levels)
    
    def extract_features(self):
        """Extract features using the model"""
        with torch.no_grad():
            self.features(self.input_tensor)
        
        # Extract the activation map from the target layer
        self.activation_map = self.activation[self.layer_name].squeeze(0)
        
        # Apply SPP layer
        self.spp_features = self.spp(self.activation_map.unsqueeze(0))
        return self.activation_map
    
    def load_image(self, image_path):
        """
        Load and preprocess an image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert("RGB")
        self.input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return self.input_tensor
    
    def show_original_image(self):
        """Display the original image"""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.input_tensor.squeeze().permute(1, 2, 0).numpy())
        plt.title("Original Image")
        plt.show()
    
    def _create_hook_fn(self, layer_name):
        """Create a hook function for a specific layer"""
        def hook_fn(module, input, output):
            self.activation[layer_name] = output.detach()
        return hook_fn
    
    def analyze_all_conv_layers(self, image_path, k=50):
        """
        Analyze all convolutional layers of AlexNet and find top activations for each
        
        Args:
            image_path (str): Path to the image
            k (int): Number of top activations to find per layer
            
        Returns:
            dict: Dictionary with results for each conv layer
        """
        # Reset activation dictionary
        self.activation = {}
        
        # Set up hooks for all conv layers
        hooks = []
        for layer_name, idx in self.conv_layers.items():
            hooks.append(
                self.features[idx].register_forward_hook(
                    self._create_hook_fn(layer_name)
                )
            )
        
        # Load the image
        self.load_image(image_path)
        
        # Forward pass to get activations
        with torch.no_grad():
            self.features(self.input_tensor)
        
        # Process each layer's activations
        results = {}
        for layer_name in self.conv_layers.keys():
            # Get the activation map for this layer
            activation_map = self.activation[layer_name].squeeze(0)
            
            # Find maximum across channels
            max_channel = activation_map.max(dim=0)[0]
            
            # Get top k activations
            flattened = max_channel.flatten()
            # Determine k value for this layer based on layer number
            layer_num = int(layer_name.replace('conv', ''))
            layer_k = max(1, k // layer_num)  # Ensure k is at least 1
            top_values, top_indices = torch.topk(flattened, k=layer_k)
            
            # Convert flat indices to 2D coordinates
            h, w = max_channel.shape
            top_coords = [(idx // w, idx % w) for idx in top_indices.tolist()]
            
            # Scale coordinates to original image size
            orig_width, orig_height = 224, 224
            activation_map_size = activation_map.shape[1:]
            
            top_coords_scaled = [
                (int((y / activation_map_size[0]) * orig_height),
                 int((x / activation_map_size[1]) * orig_width))
                for (y, x) in top_coords
            ]
            
            # Store results for this layer
            results[layer_name] = {
                'activation_map': max_channel,
                'top_values': top_values,
                'top_coords': top_coords,
                'top_coords_scaled': top_coords_scaled,
                'map_shape': activation_map_size
            }
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return results
    
    def visualize_all_layers(self, results, original_image=True):
        """
        Visualize activation maps from all layers
        
        Args:
            results (dict): Results from analyze_all_conv_layers
            original_image (bool): Whether to show the original image with activations
        """
        # If requested, show original image first
        if original_image:
            self.show_original_image()
        
        # Create a figure with subplots for each layer
        fig, axes = plt.subplots(2, len(results), figsize=(15, 8))
        
        for i, (layer_name, layer_data) in enumerate(results.items()):
            # Plot activation map
            axes[0, i].imshow(layer_data['activation_map'].numpy(), cmap='jet')
            axes[0, i].set_title(f"{layer_name} Activation Map")
            
            # Plot original image with top activations
            axes[1, i].imshow(self.input_tensor.squeeze().permute(1, 2, 0).numpy())
            for y, x in layer_data['top_coords_scaled']:
                axes[1, i].scatter(x, y, color='red', marker='x', s=20)
            axes[1, i].set_title(f"{layer_name} Top Activations")
            
        plt.tight_layout()
        plt.show()


def main():
    # Setup GUI for file selection
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    print("Select an image file for analysis:")
    image_path = filedialog.askopenfilename()
    
    if not image_path:
        print("No image selected, exiting.")
        return
    
    print("Choose analysis mode:")
    print("1. Single Layer Analysis")
    print("2. Multi-Layer Analysis")
    # mode = input("Enter choice (1 or 2): ")
    mode = 2
    if mode == "1":
        # Single layer analysis
        layer_choices = {
            "1": ("conv1", 0),
            "2": ("conv2", 3),
            "3": ("conv3", 6),
            "4": ("conv4", 8),
            "5": ("conv5", 10)
        }
        
        print("Select convolutional layer:")
        for key, (name, _) in layer_choices.items():
            print(f"{key}. {name}")
        
        layer_choice = input("Enter choice (1-5): ")
        if layer_choice in layer_choices:
            layer_name, layer_idx = layer_choices[layer_choice]
            visualizer = SingleLayerVisualizer(target_layer=layer_name, layer_index=layer_idx)
            k = int(input("Number of top activations to find (default 100): ") or 100)
            visualizer.analyze_image(image_path, k=k)
        else:
            print("Invalid layer choice. Using default conv2.")
            visualizer = SingleLayerVisualizer()
            visualizer.analyze_image(image_path)
    else:
        # Multi-layer analysis
        visualizer = MultiLayerVisualizer()
        # k = int(input("Number of top activations to find per layer (default 50): ") or 50)
        k= 50
        all_layer_results = visualizer.analyze_all_conv_layers(image_path, k=k)
        visualizer.visualize_all_layers(all_layer_results)  


if __name__ == "__main__":
    main()