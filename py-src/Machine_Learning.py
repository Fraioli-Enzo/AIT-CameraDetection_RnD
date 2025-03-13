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
    def noise_removal(image, patch_size=10, h=10):
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
    def visualize_features(image, features):
        """
        Display the input image and extracted features.

        Parameters:
        - image: Input image (numpy array).
        - features: Extracted features (torch tensor).
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
            
            # Analyze pattern structure if there are enough peaks
            if len(peak_coords) >= 2:
                # Calculate distances between consecutive peaks
                distances = np.sqrt(np.sum(np.diff(peak_coords, axis=0)**2, axis=1))
                
                # Store pattern information in the results dictionary
                results['pattern_info'][i] = {
                    'avg_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances)),
                    'regularity': float(1.0 / (1.0 + np.std(distances)/np.mean(distances) if np.mean(distances) > 0 else 1.0)),
                    'num_peaks': int(len(peak_coords))
                }
                
                # Convert numpy arrays to lists for JSON serialization
                if i == 0:  # Only need to do this once
                    # Convert the main activation maps to a serializable format
                    results['activation_maps'] = results['activation_maps'].tolist()
                
                # Convert peak coordinates to a serializable format
                for peak_loc in results['peak_locations']:
                    if isinstance(peak_loc['coordinates'], np.ndarray):
                        peak_loc['coordinates'] = peak_loc['coordinates'].tolist()
        
        return results
    
    @staticmethod
    def save_as_json(data, filename):
        """
        Save the input data as a JSON file.

        Parameters:
        - data: Input data (dictionary).
        - filename: Output JSON file name.
        """
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    # @staticmethod
    # def feature_activation_maps(features):
    #     """
    #     Generate activation maps from the extracted features and detect peaks
    #     to identify repetitive pattern structures.

    #     Parameters:
    #     - features: Extracted features (torch tensor).

    #     Returns:
    #     - Dictionary with activation maps and detected pattern information.
    #     """
    #     # Convert features to numpy for processing
    #     activation_maps = features.squeeze().cpu().numpy()
        
    #     # Get the number of feature maps
    #     num_maps = activation_maps.shape[0]
        
    #     # Store results
    #     results = {
    #         'activation_maps': activation_maps,
    #         'peak_locations': [],
    #         'pattern_info': {}
    #     }
        
    #     # Process each activation map to find peaks
    #     for i in range(num_maps):
    #         # Get current feature map
    #         feature_map = activation_maps[i]
            
    #         # Apply threshold to highlight strong activations
    #         threshold = np.mean(feature_map) + 1.5 * np.std(feature_map)
            
    #         # Find peaks (local maxima)
    #         # Using simple dilation method to identify local maxima
    #         kernel = np.ones((3, 3), np.uint8)
    #         dilated = cv2.dilate(feature_map, kernel)
    #         peaks = (feature_map == dilated) & (feature_map > threshold)
    #         peak_coords = np.column_stack(np.where(peaks))
            
    #         if len(peak_coords) > 0:
    #             results['peak_locations'].append({
    #                 'map_index': i,
    #                 'coordinates': peak_coords
    #             })
                
    #             # Analyze pattern structure
    #             if len(peak_coords) >= 2:
    #                 # Calculate distances between consecutive peaks
    #                 distances = np.sqrt(np.sum(np.diff(peak_coords, axis=0)**2, axis=1))
                    
    #                 results['pattern_info'][i] = {
    #                     'avg_distance': np.mean(distances),
    #                     'std_distance': np.std(distances),
    #                     'regularity': 1.0 / (1.0 + np.std(distances)/np.mean(distances) if np.mean(distances) > 0 else 1.0),
    #                     'num_peaks': len(peak_coords)
    #                 }
        
    #     return results


# class PeriodicDetection:
#     @staticmethod
#     def detect_periodic_patterns(feature_results):
#         """
#         Detect periodic patterns from feature activation maps.
        
#         Parameters:
#         - feature_results: Results from feature extraction containing activation maps and peak locations
        
#         Returns:
#         - Dictionary containing periodic pattern information
#         """
#         pattern_info = {}
        
#         # Process each feature map with detected peaks
#         for peak_data in feature_results['peak_locations']:
#             map_idx = peak_data['map_index']
#             coords = peak_data['coordinates']
            
#             # Need at least 3 peaks to detect periodicity
#             if len(coords) < 3:
#                 continue
                
#             # Calculate displacement vectors between all pairs of peaks
#             displacement_vectors = []
#             for i in range(len(coords)):
#                 for j in range(i + 1, len(coords)):
#                     vector = coords[j] - coords[i]
#                     # Store as tuple to make it hashable for counting
#                     displacement_vectors.append(tuple(vector))
            
#             # Hough voting mechanism to find most common displacement
#             vector_counts = {}
#             for vector in displacement_vectors:
#                 # Group similar vectors (with small differences)
#                 matched = False
#                 for key_vector in vector_counts.keys():
#                     # Check if vectors are similar (within threshold)
#                     if np.sqrt(sum((a-b)**2 for a, b in zip(vector, key_vector))) < 2:
#                         vector_counts[key_vector] += 1
#                         matched = True
#                         break
                
#                 if not matched:
#                     vector_counts[vector] = 1
            
#             # Find the most frequent displacement vector
#             if vector_counts:
#                 dominant_vector = max(vector_counts.items(), key=lambda x: x[1])
                
#                 # Calculate periodicity metrics
#                 pattern_info[map_idx] = {
#                     'dominant_vector': dominant_vector[0],
#                     'vector_frequency': dominant_vector[1],
#                     'confidence': dominant_vector[1] / len(displacement_vectors),
#                     'pattern_size': np.sqrt(sum(v**2 for v in dominant_vector[0])),
#                     'direction': np.arctan2(dominant_vector[0][1], dominant_vector[0][0]) * 180 / np.pi
#                 }
        
#         return {
#             'periodic_patterns': pattern_info,
#             'num_periodic_patterns': len(pattern_info),
#             'avg_pattern_size': np.mean([info['pattern_size'] for info in pattern_info.values()]) if pattern_info else 0
#         }

# class PeriodicSegmentation:
#     @staticmethod
#     def segment_periodic_patterns(image, pattern_info):
#         """
#         Segment periodic patterns in the image using the detected pattern information.
        
#         Parameters:
#         - image: Input image (numpy array)
#         - pattern_info: Dictionary containing information about detected periodic patterns
        
#         Returns:
#         - Segmented image with highlighted periodic patterns
#         - Dictionary with segmentation details
#         """
#         # Create output image for visualization
#         segmented_image = image.copy()
#         segmentation_results = {
#             'primitives': [],
#             'patterns': []
#         }
        
#         # Skip if no patterns detected
#         if not pattern_info['periodic_patterns']:
#             return segmented_image, segmentation_results
        
#         # Generate mask for segmentation
#         mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
#         # Process each detected pattern
#         for pattern_id, info in pattern_info['periodic_patterns'].items():
#             # Extract pattern properties
#             dominant_vector = info['dominant_vector']
#             pattern_size = info['pattern_size']
#             confidence = info['confidence']
            
#             # Only process high-confidence patterns
#             if confidence < 0.5:
#                 continue
            
#             # Calculate primitive size based on pattern vector
#             primitive_size = max(int(pattern_size * 0.8), 3)
            
#             # Iterative Projection and Matching (IPM)
#             # 1. Initialize primitive template
#             x_center = image.shape[1] // 2
#             y_center = image.shape[0] // 2
            
#             # Extract initial primitive estimate from center of image
#             primitive_template = image[
#                 max(0, y_center - primitive_size):min(image.shape[0], y_center + primitive_size),
#                 max(0, x_center - primitive_size):min(image.shape[1], x_center + primitive_size)
#             ]
            
#             # 2. Iterative refinement
#             for iteration in range(3):  # 3 iterations of IPM
#                 # Apply template matching to find instances
#                 matching_result = cv2.matchTemplate(
#                     image, 
#                     primitive_template, 
#                     cv2.TM_CCOEFF_NORMED
#                 )
                
#                 # Find peaks in matching result
#                 threshold = 0.6 + (iteration * 0.1)  # Increase threshold with iterations
#                 loc = np.where(matching_result >= threshold)
                
#                 # Store primitive locations
#                 primitive_locations = list(zip(*loc[::-1]))
                
#                 # Add primitives to the mask
#                 h, w = primitive_template.shape[:2]
#                 for pt in primitive_locations:
#                     # Draw rectangle around each primitive
#                     cv2.rectangle(segmented_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
#                     cv2.rectangle(mask, pt, (pt[0] + w, pt[1] + h), 255, -1)
                    
#                     segmentation_results['primitives'].append({
#                         'x': int(pt[0]),
#                         'y': int(pt[1]),
#                         'width': w,
#                         'height': h,
#                         'confidence': float(matching_result[pt[1], pt[0]])
#                     })
                
#                 # Refine primitive template by averaging matched regions
#                 if primitive_locations:
#                     # Extract all matched regions
#                     matched_regions = []
#                     for pt in primitive_locations[:min(10, len(primitive_locations))]:  # Limit to 10 highest matches
#                         x, y = pt
#                         if (y + h <= image.shape[0] and x + w <= image.shape[1]):
#                             region = image[y:y+h, x:x+w]
#                             matched_regions.append(region)
                    
#                     # Average the matched regions to refine template
#                     if matched_regions:
#                         primitive_template = np.mean(matched_regions, axis=0).astype(np.uint8)
            
#             # Add pattern info to results
#             segmentation_results['patterns'].append({
#                 'pattern_id': int(pattern_id),
#                 'vector': [float(v) for v in dominant_vector],
#                 'confidence': float(confidence),
#                 'primitive_count': len(primitive_locations)
#             })
        
#         # Apply mask to highlight all periodic regions
#         periodic_regions = cv2.bitwise_and(image, image, mask=mask)
        
#         # Overlay segmentation on original image
#         alpha = 0.7
#         segmented_image = cv2.addWeighted(segmented_image, alpha, periodic_regions, 1-alpha, 0)
        
#         return segmented_image, segmentation_results

# class PostProcessing:
#     @staticmethod

class Pipeline:
    @staticmethod
    def process_pattern(image_path):  
        # Load the image
        image = DataProcessing.image_loader(image_path)

        # Preprocessing image
        denoised_image = DataProcessing.noise_removal(image)
        DataProcessing.visualize_image(denoised_image)

        # Feature extraction
        model = FeatureExtraction.CNN_model_selection()
        features = FeatureExtraction.features_extraction(model, denoised_image)
        print(features.shape)
        print(features)
        result = FeatureExtraction.visualize_features(image, features)
        FeatureExtraction.save_as_json(result, "results.json")
        print(result)


   
def main():
    # Load the image
    tk.Tk().withdraw()
    image_path = filedialog.askopenfilename()
    # image_path = "Images/Capture1.PNG"
    Pipeline.process_pattern(image_path)

    
if __name__ == "__main__":
    main()
