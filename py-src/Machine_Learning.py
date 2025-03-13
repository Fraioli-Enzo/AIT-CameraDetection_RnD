import cv2
import tkinter as tk
from tkinter import filedialog
from torchvision.models import alexnet
from torch import nn
import numpy as np
import torch
import json
import os

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
    
    @staticmethod
    def feature_activation_maps(features):
        """
        Generate activation maps from the extracted features and detect peaks
        to identify repetitive pattern structures.

        Parameters:
        - features: Extracted features (torch tensor).

        Returns:
        - Dictionary with activation maps and detected pattern information.
        """
        # Convert features to numpy for processing
        activation_maps = features.squeeze().cpu().numpy()
        
        # Get the number of feature maps
        num_maps = activation_maps.shape[0]
        
        # Store results
        results = {
            'activation_maps': activation_maps,
            'peak_locations': [],
            'pattern_info': {}
        }
        
        # Process each activation map to find peaks
        for i in range(num_maps):
            # Get current feature map
            feature_map = activation_maps[i]
            
            # Apply threshold to highlight strong activations
            threshold = np.mean(feature_map) + 1.5 * np.std(feature_map)
            
            # Find peaks (local maxima)
            # Using simple dilation method to identify local maxima
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(feature_map, kernel)
            peaks = (feature_map == dilated) & (feature_map > threshold)
            peak_coords = np.column_stack(np.where(peaks))
            
            if len(peak_coords) > 0:
                results['peak_locations'].append({
                    'map_index': i,
                    'coordinates': peak_coords
                })
                
                # Analyze pattern structure
                if len(peak_coords) >= 2:
                    # Calculate distances between consecutive peaks
                    distances = np.sqrt(np.sum(np.diff(peak_coords, axis=0)**2, axis=1))
                    
                    results['pattern_info'][i] = {
                        'avg_distance': np.mean(distances),
                        'std_distance': np.std(distances),
                        'regularity': 1.0 / (1.0 + np.std(distances)/np.mean(distances) if np.mean(distances) > 0 else 1.0),
                        'num_peaks': len(peak_coords)
                    }
        
        return results


class PeriodicDetection:
    @staticmethod
    def detect_periodic_patterns(feature_results):
        """
        Detect periodic patterns from feature activation maps.
        
        Parameters:
        - feature_results: Results from feature extraction containing activation maps and peak locations
        
        Returns:
        - Dictionary containing periodic pattern information
        """
        pattern_info = {}
        
        # Process each feature map with detected peaks
        for peak_data in feature_results['peak_locations']:
            map_idx = peak_data['map_index']
            coords = peak_data['coordinates']
            
            # Need at least 3 peaks to detect periodicity
            if len(coords) < 3:
                continue
                
            # Calculate displacement vectors between all pairs of peaks
            displacement_vectors = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    vector = coords[j] - coords[i]
                    # Store as tuple to make it hashable for counting
                    displacement_vectors.append(tuple(vector))
            
            # Hough voting mechanism to find most common displacement
            vector_counts = {}
            for vector in displacement_vectors:
                # Group similar vectors (with small differences)
                matched = False
                for key_vector in vector_counts.keys():
                    # Check if vectors are similar (within threshold)
                    if np.sqrt(sum((a-b)**2 for a, b in zip(vector, key_vector))) < 2:
                        vector_counts[key_vector] += 1
                        matched = True
                        break
                
                if not matched:
                    vector_counts[vector] = 1
            
            # Find the most frequent displacement vector
            if vector_counts:
                dominant_vector = max(vector_counts.items(), key=lambda x: x[1])
                
                # Calculate periodicity metrics
                pattern_info[map_idx] = {
                    'dominant_vector': dominant_vector[0],
                    'vector_frequency': dominant_vector[1],
                    'confidence': dominant_vector[1] / len(displacement_vectors),
                    'pattern_size': np.sqrt(sum(v**2 for v in dominant_vector[0])),
                    'direction': np.arctan2(dominant_vector[0][1], dominant_vector[0][0]) * 180 / np.pi
                }
        
        return {
            'periodic_patterns': pattern_info,
            'num_periodic_patterns': len(pattern_info),
            'avg_pattern_size': np.mean([info['pattern_size'] for info in pattern_info.values()]) if pattern_info else 0
        }



class Pipeline:
    @staticmethod
    def process_pattern(image_path):  
        image = DataProcessing.image_loader(image_path)
        denoised_image = DataProcessing.noise_removal(image)
        DataProcessing.visualize_image(denoised_image)
        model = FeatureExtraction.CNN_model_selection()
        features = FeatureExtraction.features_extraction(model, denoised_image)
        print(features)
        results = FeatureExtraction.feature_activation_maps(features)
###################################################################     
        # Save results to a JSON file
        # Extract pattern information for JSON serialization
        pattern_data = {
            'num_activation_maps': len(results['activation_maps']),
            'num_peak_maps': len(results['peak_locations']),
            'pattern_info': results['pattern_info']
        }

        # Create output directory if it doesn't exist
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename based on input image
        base_filename = os.path.basename(image_path)
        output_filename = os.path.join(output_dir, f"{os.path.splitext(base_filename)[0]}_analysis.json")

        # Save to JSON file
        with open(output_filename, 'w') as f:
            json.dump(pattern_data, f, indent=4)

        print(f"Results saved to {output_filename}")
###################################################################

        # Detect periodic patterns
        periodic_results = PeriodicDetection.detect_periodic_patterns(results)
        print(periodic_results)






def main():
    # Load the image
    tk.Tk().withdraw()
    # image_path = filedialog.askopenfilename()
    image_path = "Images/Capture.PNG"
    Pipeline.process_pattern(image_path)

    
if __name__ == "__main__":
    main()
