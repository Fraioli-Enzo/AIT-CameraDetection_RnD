import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class MLConfig:
    """Configuration for the machine learning defect detection."""
    # Feature extraction parameters
    patch_size: int = 64  # Size of patches to extract
    stride: int = 32      # Step size for sliding window
    # PCA parameters
    n_components: int = 50  # Number of PCA components to keep
    # Isolation Forest parameters
    n_estimators: int = 100
    contamination: float = 0.1
    random_state: int = 42
    # Defect detection parameters
    threshold: float = -0.5  # Anomaly score threshold (adjust based on validation)

class FeatureExtractor:
    """Extract features from fabric images for defect detection."""
    
    @staticmethod
    def extract_patches(image: np.ndarray, patch_size: int, stride: int) -> List[np.ndarray]:
        """Extract overlapping patches from the image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        patches = []
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = gray[y:y + patch_size, x:x + patch_size]
                patches.append(patch)
        
        return patches
    
    @staticmethod
    def extract_features(image: np.ndarray, config: MLConfig) -> np.ndarray:
        """Extract texture and statistical features from the image."""
        patches = FeatureExtractor.extract_patches(image, config.patch_size, config.stride)
        features = []
        
        for patch in patches:
            # Flatten the patch
            flattened = patch.flatten()
            
            # Add statistical features
            features.append(np.concatenate([
                flattened,
                [patch.mean(), patch.std(), patch.min(), patch.max()]
            ]))
            
        return np.array(features)

class FabricDefectDetector:
    """Machine learning model for fabric defect detection."""
    
    def __init__(self, config: MLConfig):
        """Initialize the defect detector with configuration parameters."""
        self.config = config
        self.pca = PCA(n_components=config.n_components)
        self.model = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=config.random_state
        )
        self.is_trained = False
    
    def train(self, reference_images: List[np.ndarray]) -> None:
        """Train the model on defect-free reference images."""
        # Extract features from all reference images
        all_features = []
        for image in reference_images:
            features = FeatureExtractor.extract_features(image, self.config)
            all_features.append(features)
        
        # Combine all features
        X = np.vstack(all_features)
        
        # Perform dimensionality reduction with PCA
        print(f"Performing PCA on {X.shape[0]} samples with {X.shape[1]} features...")
        X_pca = self.pca.fit_transform(X)
        print(f"Reduced to {X_pca.shape[1]} features, explained variance: {np.sum(self.pca.explained_variance_ratio_):.2f}")
        
        # Train the anomaly detection model
        print("Training anomaly detection model...")
        self.model.fit(X_pca)
        self.is_trained = True
        print("Model training complete.")
    
    def detect_defects(self, test_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect defects in a test image.
        
        Returns:
            defect_map: A map showing detected defects
            anomaly_score: Overall anomaly score for the image
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before defect detection.")
        
        # Extract features from test image
        features = FeatureExtractor.extract_features(test_image, self.config)
        
        # Apply PCA transformation
        features_pca = self.pca.transform(features)
        
        # Get anomaly scores (-1 for anomalies, 1 for normal samples)
        anomaly_scores = self.model.decision_function(features_pca)
        predictions = self.model.predict(features_pca)  # -1 for anomalies, 1 for normal
        
        # Convert anomaly scores to defect map
        defect_map = self._create_defect_map(test_image, predictions, anomaly_scores)
        
        # Calculate overall anomaly score (lower is more anomalous)
        overall_score = np.mean(anomaly_scores)
        
        return defect_map, overall_score
    
    def _create_defect_map(self, image: np.ndarray, predictions: np.ndarray, 
                          scores: np.ndarray) -> np.ndarray:
        """
        Create a heatmap of defects from patch predictions.
        
        Red areas indicate potential defects (anomalous regions).
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create a color map for visualization
        height, width = gray.shape
        defect_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Convert grayscale to BGR for visualization
        defect_map = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Track which patches were marked as anomalies
        patch_idx = 0
        for y in range(0, height - self.config.patch_size + 1, self.config.stride):
            for x in range(0, width - self.config.patch_size + 1, self.config.stride):
                if patch_idx < len(predictions):
                    # Use anomaly score to determine color intensity
                    score = scores[patch_idx]
                    
                    # Mark patches with low scores (anomalies) with a red overlay
                    if score < self.config.threshold:
                        # Draw a rectangle with red color (severity based on score)
                        intensity = int(max(0, min(255, 255 * (1 - (score - self.config.threshold)))))
                        cv2.rectangle(
                            defect_map, 
                            (x, y), 
                            (x + self.config.patch_size, y + self.config.patch_size),
                            (0, 0, intensity),  # Red color with intensity based on anomaly score
                            2
                        )
                    patch_idx += 1
        
        return defect_map
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        model_data = {
            'pca': self.pca,
            'model': self.model,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'FabricDefectDetector':
        """Load a saved model from a file."""
        model_data = joblib.load(filepath)
        
        detector = cls(model_data['config'])
        detector.pca = model_data['pca']
        detector.model = model_data['model']
        detector.is_trained = True
        
        return detector

class GUI:
    """Simple GUI for the fabric defect detection system."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Fabric Defect Detection System")
        self.root.geometry("500x400")
        
        self.config = MLConfig()
        self.detector = FabricDefectDetector(self.config)
        self.reference_images = []
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI elements."""
        # Title
        tk.Label(self.root, text="Fabric Defect Detection", font=("Arial", 16)).pack(pady=10)
        
        # Training section
        training_frame = tk.LabelFrame(self.root, text="Training")
        training_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(training_frame, text="Select Reference Images", command=self._load_reference_images).pack(pady=5)
        self.training_status = tk.Label(training_frame, text="Status: Not trained")
        self.training_status.pack(pady=5)
        tk.Button(training_frame, text="Train Model", command=self._train_model).pack(pady=5)
        
        # Model save/load
        model_frame = tk.LabelFrame(self.root, text="Model Management")
        model_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(model_frame, text="Save Model", command=self._save_model).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(model_frame, text="Load Model", command=self._load_model).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Testing section
        testing_frame = tk.LabelFrame(self.root, text="Defect Detection")
        testing_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(testing_frame, text="Select Test Image", command=self._detect_defects).pack(pady=5)
        self.detection_result = tk.Label(testing_frame, text="Status: No detection performed")
        self.detection_result.pack(pady=5)
    
    def _load_reference_images(self):
        """Load multiple reference (defect-free) images."""
        filepaths = filedialog.askopenfilenames(
            title="Select Reference Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not filepaths:
            return
        
        self.reference_images = []
        for path in filepaths:
            img = cv2.imread(path)
            if img is not None:
                self.reference_images.append(img)
        
        self.training_status.config(text=f"Status: {len(self.reference_images)} reference images loaded")
    
    def _train_model(self):
        """Train the model using loaded reference images."""
        if not self.reference_images:
            messagebox.showerror("Error", "No reference images loaded")
            return
        
        try:
            self.detector.train(self.reference_images)
            self.training_status.config(text=f"Status: Model trained on {len(self.reference_images)} images")
        except Exception as e:
            messagebox.showerror("Training Error", f"Error during training: {e}")
    
    def _save_model(self):
        """Save the trained model."""
        if not self.detector.is_trained:
            messagebox.showerror("Error", "No trained model to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".joblib",
            filetypes=[("Joblib files", "*.joblib")]
        )
        
        if filepath:
            try:
                self.detector.save_model(filepath)
            except Exception as e:
                messagebox.showerror("Save Error", f"Error saving model: {e}")
    
    def _load_model(self):
        """Load a trained model."""
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Joblib files", "*.joblib")]
        )
        
        if filepath:
            try:
                self.detector = FabricDefectDetector.load_model(filepath)
                self.training_status.config(text="Status: Model loaded from file")
            except Exception as e:
                messagebox.showerror("Load Error", f"Error loading model: {e}")
    
    def _detect_defects(self):
        """Detect defects in a selected test image."""
        if not self.detector.is_trained:
            messagebox.showerror("Error", "Model not trained")
            return
        
        filepath = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not filepath:
            return
        
        test_image = cv2.imread(filepath)
        if test_image is None:
            messagebox.showerror("Error", "Could not read image")
            return
        
        try:
            # Detect defects
            defect_map, anomaly_score = self.detector.detect_defects(test_image)
            
            # Update status
            result = "DEFECT DETECTED" if anomaly_score < self.config.threshold else "NO DEFECT DETECTED"
            self.detection_result.config(text=f"Result: {result} (Score: {anomaly_score:.3f})")
            
            # Display images
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title(f"Defect Map (Score: {anomaly_score:.3f})")
            plt.imshow(cv2.cvtColor(defect_map, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Error during defect detection: {e}")


def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()