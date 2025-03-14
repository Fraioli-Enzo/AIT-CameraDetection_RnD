import torch
import torchvision.models as models
from torchvision.models import AlexNet_Weights
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

class FeatureMapVisualizer:
    def __init__(self, target_layer='conv2', layer_index=3):
        """
        Initialize the feature map visualizer with the AlexNet model
        
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


def main():
    # Create visualizer instance
    visualizer = FeatureMapVisualizer(target_layer='conv2', layer_index=3)
    
    # Analyze image
    image_path = "Images/Capture.png"  # Change this to your image path
    visualizer.analyze_image(image_path, k=100)

if __name__ == "__main__":
    main()