import numpy as np
import json
import os
from plyfile import PlyData, PlyElement
import torch
import json
import torch
import os

from gsworld.constants import ASSET_DIR
from gsworld.mani_skill.utils.wrappers import Semantic3DGSWrapper

class GaussianModelMerger:
    """
    A class to load, manage, and merge multiple Gaussian models from PLY files
    using a JSON configuration file.
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize the GaussianModelMerger.
        """
        self.device = device
        self.models = []
        self.model_paths = []
        self.model_configs = []
        self.merged_model = None
        
    def load_config_from_json(self, json_path):
        """
        Load model configurations from a JSON configuration file.
        Reads all keys and values from the JSON.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON configuration file not found: {json_path}")
            
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
                
            # Print all top-level keys in the JSON
            print(f"JSON contains the following keys: {list(config.keys())}")
            
            # Extract model configurations
            model_configs = []
            
            if "models" in config and isinstance(config["models"], list):
                print(f"Found {len(config['models'])} model entries in the JSON")
                
                for i, model_entry in enumerate(config["models"]):
                    # Print all keys for this model entry
                    print(f"Model {i} contains keys: {list(model_entry.keys())}")
                    
                    # Store the complete model configuration
                    model_configs.append(model_entry)
            else:
                raise ValueError("JSON file should contain a 'models' list")
                
            # Store model configurations
            self.model_configs = model_configs
            
            return model_configs
        
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {json_path}")
        
    def assign_semantic_labels(self, model_data, semantic_labels):
        """
        Assign semantic labels to the model data.
        
        Args:
            model_data: The Gaussian model to assign labels to
            semantic_labels: Can be a path to a numpy file, an integer label,
                            or a tensor of per-point labels
        
        Returns:
            The model with semantic labels assigned
        """
        if isinstance(semantic_labels, str):
            assert os.path.exists(semantic_labels)
            # Load labels from file
            labels = np.load(semantic_labels)
            model_data._semantics = torch.from_numpy(labels).to(self.device)[..., None]
            print(f"Loaded semantic labels from {semantic_labels}, shape: {model_data._semantics.shape}")
        elif isinstance(semantic_labels, (int, float)):
            # Assign a single label to all points
            model_data._semantics = torch.zeros(model_data._xyz.shape[0], 1, device=self.device)
            model_data._semantics = torch.full_like(model_data._semantics, semantic_labels)
            print(f"Assigned single semantic label {semantic_labels} to all points")
        elif isinstance(semantic_labels, torch.Tensor):
            # Use provided tensor directly
            model_data._semantics = semantic_labels.to(self.device)
        else:
            # Default: all zeros
            model_data._semantics = torch.zeros(model_data._xyz.shape[0], 1, device=self.device)
            print("No valid semantic labels provided, defaulting to zeros")
        
        return model_data
    
    def apply_transformation(self, model, transformation_matrix):
        """
        Apply a 4x4 transformation matrix to a Gaussian model.
        
        Args:
            model: The Gaussian model
            transformation_matrix: A 4x4 transformation matrix (list/array)
            
        Returns:
            The transformed model
        """
        if transformation_matrix is None:
            return model
            
        try:
            # Convert transformation to PyTorch tensor
            if isinstance(transformation_matrix, list):
                if len(transformation_matrix) == 16:
                    # Reshape from flat list to 4x4 matrix
                    transformation_matrix = torch.tensor(transformation_matrix, 
                                                        device=self.device).reshape(4, 4).float()
                else:
                    raise ValueError(f"Transformation matrix should have 16 elements, got {len(transformation_matrix)}")
            elif isinstance(transformation_matrix, np.ndarray):
                transformation_matrix = torch.from_numpy(transformation_matrix).to(self.device).float()
            
            # Extract rotation and translation from the transformation matrix
            rotation = transformation_matrix[:3, :3]
            translation = transformation_matrix[:3, 3]
            
            # Get the current point positions
            xyz = model.get_xyz()
            
            # Apply rotation
            rotated_xyz = torch.matmul(xyz, rotation.T)
            
            # Apply translation
            transformed_xyz = rotated_xyz + translation.unsqueeze(0)
            
            # Update the model's positions
            model.set_xyz(transformed_xyz)
            
            # Also handle rotation of the Gaussian orientations
            # Convert quaternion to rotation matrix and compose with transformation
            rotation_quat = model.get_rotation()
            # TODO: Implement proper quaternion rotation composition
            # For now, just update position
            
            print(f"Applied transformation matrix")
            
            return model
        except Exception as e:
            print(f"Error applying transformation: {str(e)}")
            return model
    
    def load_models_from_config(self, json_path):
        """
        Load models from a JSON configuration file.
        """
        model_configs = self.load_config_from_json(json_path)
        return self.load_multiple_models(model_configs)
    
    def load_model_from_config(self, model_config):
        """
        Load a model from a configuration dictionary.
        """
        if "data_path" not in model_config:
            raise ValueError(f"Missing required 'data_path' in model config")
        
        ply_path = os.path.join(ASSET_DIR, model_config["data_path"])
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
        
        # Load semantic labels if specified
        semantic_labels = model_config.get("semantic_labels", None)
        
        # Load transformation if specified
        transformation = model_config.get("transformation", None)
            
        # Load the model
        model = Semantic3DGSWrapper(3)
        model.load_ply(ply_path)
        
        # Apply semantic labels
        if semantic_labels is not None:
            if isinstance(semantic_labels, str):
                semantic_labels = os.path.join(ASSET_DIR, semantic_labels)
            model = self.assign_semantic_labels(model, semantic_labels)
        
        self.models.append(model)
        self.model_paths.append(ply_path)
        return len(self.models) - 1
    
    def load_multiple_models(self, model_configs):
        """
        Load multiple models from configuration dictionaries.
        """
        indices = []
        for config in model_configs:
            idx = self.load_model_from_config(config)
            indices.append(idx)
            print(f"Loaded model from {config['data_path']}")
        return indices
    
    def get_model(self, index):
        """
        Get a model by its index.
        """
        if 0 <= index < len(self.models):
            return self.models[index]
        else:
            raise IndexError(f"Model index {index} is out of range")
    
    def merge_models(self, indices=None):
        """
        Merge selected models into one. If no indices are provided, merge all models.
        """
        if not self.models:
            raise ValueError("No models to merge")
            
        # Select models to merge
        if indices is None:
            models_to_merge = self.models
            paths_to_merge = self.model_paths
        else:
            models_to_merge = [self.get_model(idx) for idx in indices]
            paths_to_merge = [self.model_paths[idx] for idx in indices]
            
        if not models_to_merge:
            raise ValueError("No valid models to merge")
            
        print(f"Merging {len(models_to_merge)} models: {', '.join(paths_to_merge)}")
        
        # Create a new model for the merged result
        merged_model = Semantic3DGSWrapper(3)
        
        # Initialize lists to store combined attributes
        all_xyz = []
        all_features_dc = []
        all_features_rest = []
        all_scaling = []
        all_rotation = []
        all_opacity = []
        all_semantics = []
        
        # Collect data from all models
        for model in models_to_merge:
            # Get the model's attributes
            all_xyz.append(model._xyz)
            all_features_dc.append(model._features_dc)
            all_features_rest.append(model._features_rest)
            all_scaling.append(model._scaling)
            all_rotation.append(model._rotation)
            all_opacity.append(model._opacity)
            
            if hasattr(model, '_semantics'):
                all_semantics.append(model._semantics)
            else:
                # Create default semantics if not available
                all_semantics.append(torch.zeros(model._xyz.shape[0], 1, device=self.device))
        
        # Concatenate all attributes
        merged_model._xyz = torch.cat(all_xyz, dim=0)
        merged_model._features_dc = torch.cat(all_features_dc, dim=0)
        merged_model._features_rest = torch.cat(all_features_rest, dim=0)
        merged_model._scaling = torch.cat(all_scaling, dim=0)
        merged_model._rotation = torch.cat(all_rotation, dim=0)
        merged_model._opacity = torch.cat(all_opacity, dim=0)
        merged_model._semantics = torch.cat(all_semantics, dim=0)
        
        # Store the merged model
        self.merged_model = merged_model
        
        print("Models merged successfully!")
        return merged_model
    
    def save_merged_model(self, output_path):
        """
        Save the previously merged model to a PLY file.
        """
        if self.merged_model is None:
            raise ValueError("No merged model exists. Call merge_models() first.")
            
        self.merged_model.save_ply(output_path)
        print(f"Merged model saved to {output_path}")
        return True
    
    def get_merged_model(self):
        """
        Get the merged model.
        """
        if self.merged_model is None:
            raise ValueError("No merged model exists. Call merge_models() first.")
            
        return self.merged_model
    
    def clear_models(self):
        """
        Clear all loaded models.
        """
        self.models = []
        self.model_paths = []
        self.model_configs = []
        print("Cleared all models from memory")


# Example usage with JSON configuration
def main(path):
    
    # Initialize the merger
    merger = GaussianModelMerger(device='cuda')
    
    # Load models from JSON configuration
    indices = merger.load_models_from_config(path)
    
    # Merge all models without saving
    merged_model = merger.merge_models()
    

    merger.clear_models()
    return merged_model

if __name__ == "__main__":
    json_path = os.path.join(FILE_PATH, "../assets/franka_merger.json")
    main(json_path)