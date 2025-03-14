import torch
import torchvision.models as models
from torchvision.models import AlexNet_Weights
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Load a pre-trained AlexNet model
model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
model.eval()

# Remove the last layer (classifier) to get feature activations
features = model.features  # Extract convolutional layers only

# Hook to capture activations
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Attach the hook to the last convolutional layer
layer_name = 'conv1'  # Switch to an earlier layer
features[0].register_forward_hook(get_activation(layer_name))  

# Image preprocessing (Modify as needed for your dataset)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load an example fabric image
image_path = "Images/Capture.png"  # Change this to your image path
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
plt.imshow(input_tensor.squeeze().permute(1, 2, 0).numpy())
plt.show()

# Forward pass
with torch.no_grad():
    features(input_tensor)

# Extract the activation map from the last convolutional layer
activation_map = activation[layer_name].squeeze(0)  # Remove batch dimension

# Get the top 30 activation points across all channels
max_channel = activation_map.max(dim=0)[0]  # Collapse channels
flattened = max_channel.flatten()
top_values, top_indices = torch.topk(flattened, k=200)  # Get top 30 values and indices

# Convert flat indices to 2D coordinates
h, w = max_channel.shape
top_coords = [(idx // w, idx % w) for idx in top_indices.tolist()]

# Display top activation values and coordinates
print(f"Top 30 Activation Values: {top_values.tolist()}")
print(f"Top 30 Activation Coordinates (h, w): {top_coords}")

# Visualize the activation map with top 30 points
plt.figure(figsize=(10, 8))
plt.imshow(max_channel.numpy(), cmap='jet')
for coord in top_coords:
    plt.scatter(coord[1], coord[0], color='red', marker='x')
plt.colorbar()
plt.title("Activation Map with Top 30 Peak Points")
plt.show()

# Assuming your original image is 224x224
orig_width, orig_height = 224, 224

# Activation map size
activation_map_size = activation_map.shape[1:]  # (Height, Width)
print(f"Activation map size: {activation_map_size}")

# Scale the top coordinates to original image size
top_coords_scaled = [
    (int((y / activation_map_size[0]) * orig_height),
     int((x / activation_map_size[1]) * orig_width))
    for (y, x) in top_coords
]

print(f"Top 30 Activation Coordinates in Original Image: {top_coords_scaled}")

# Display the original image with top 30 peak points
plt.figure(figsize=(10, 8))
plt.imshow(input_tensor.squeeze().permute(1, 2, 0).numpy())
for y, x in top_coords_scaled:
    plt.scatter(x, y, color='red', marker='x', s=100)
plt.title("Original Image with Top 30 Activation Points")
plt.show()