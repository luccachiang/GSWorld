import numpy as np
from scipy.spatial import cKDTree

def transfer_labels(source_points, source_labels, target_points, transformation_matrix):
    """
    Transfer labels from a labeled point set to an unlabeled point set using nearest neighbor matching.
    
    Parameters:
    source_points: np.ndarray
        Array of shape (N, 3) containing the labeled point cloud
    source_labels: np.ndarray
        Array of shape (N,) containing the labels for source points
    target_points: np.ndarray
        Array of shape (M, 3) containing the unlabeled point cloud
    transformation_matrix: np.ndarray
        4x4 transformation matrix to align src2tgt, m@src=tgt
        
    Returns:
    np.ndarray: Array of shape (M,) containing the transferred labels for target points
    np.ndarray: Array of shape (M,) containing the distances to nearest neighbors
    """
    # Convert points to homogeneous coordinates
    target_homog = np.hstack([target_points, np.ones((len(target_points), 1))]) # N,3 -> N,4
    
    # Apply transformation to target points
    transformed_target = (np.linalg.inv(transformation_matrix) @ target_homog.T).T
    
    # Convert back to 3D coordinates
    transformed_target = transformed_target[:, :3]
    
    # Create KD-tree for efficient nearest neighbor search
    tree = cKDTree(source_points)
    
    # Find nearest neighbors
    distances, indices = tree.query(transformed_target, k=1)
    
    # Transfer labels based on nearest neighbors
    transferred_labels = source_labels[indices]
    
    return transferred_labels, distances

def point_to_bbox_distance(point, min_bound, max_bound):
    """
    Calculate the minimum distance from a point to a bounding box.
    Returns 0 if the point is inside the box.
    """
    # Calculate the distance component-wise
    dx = max(min_bound[0] - point[0], 0, point[0] - max_bound[0])
    dy = max(min_bound[1] - point[1], 0, point[1] - max_bound[1])
    dz = max(min_bound[2] - point[2], 0, point[2] - max_bound[2])
    
    # Return Euclidean distance
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def transfer_labels_with_bbox(source_points, source_labels, target_points, transformation_matrix, semantic_bboxes, bbox_distance_threshold=0.1):
    """
    Transfer labels from source to target points with flexible bounding box validation.
    
    Parameters:
    source_points: np.ndarray - Array of shape (N, 3) containing the labeled point cloud
    source_labels: np.ndarray - Array of shape (N,) containing the labels for source points
    target_points: np.ndarray - Array of shape (M, 3) containing the unlabeled point cloud
    transformation_matrix: np.ndarray - 4x4 transformation matrix to align src2tgt
    semantic_bboxes: dict - Dictionary mapping semantic index to (min_bound, max_bound)
    bbox_distance_threshold: float - Maximum allowed distance from bounding box (default: 0.1)
    
    Returns:
    np.ndarray: Array of shape (M,) containing the transferred labels, with -1 for invalid points
    np.ndarray: Array of shape (M,) containing the distances to nearest neighbors
    """
    # Convert target points to homogeneous coordinates
    target_homog = np.hstack([target_points, np.ones((len(target_points), 1))])
    
    # Transform target points to source space
    transformed_target = (np.linalg.inv(transformation_matrix) @ target_homog.T).T
    transformed_target = transformed_target[:, :3]
    
    # Find nearest neighbors
    tree = cKDTree(source_points)
    distances, indices = tree.query(transformed_target, k=1)
    
    # Get initial label transfer
    transferred_labels = source_labels[indices]
    
    # Validate points against bounding boxes with distance threshold
    for i in range(len(transformed_target)):
        point = transformed_target[i]
        current_label = transferred_labels[i]
        
        if current_label == -1:
            continue
            
        # Check distance to corresponding bbox
        if current_label in semantic_bboxes:
            min_bound, max_bound = semantic_bboxes[current_label]
            distance_to_bbox = point_to_bbox_distance(point, min_bound, max_bound)
            
            if distance_to_bbox > bbox_distance_threshold:
                # Try to find a closer bbox
                min_distance = float('inf')
                best_label = -1
                
                for sem_idx, (min_b, max_b) in semantic_bboxes.items():
                    dist = point_to_bbox_distance(point, min_b, max_b)
                    if dist < min_distance and dist <= bbox_distance_threshold:
                        min_distance = dist
                        best_label = sem_idx
                
                transferred_labels[i] = best_label
        else:
            # If label doesn't have a corresponding bbox, mark as invalid
            transferred_labels[i] = -1
    
    return transferred_labels, distances

def validate_labels(distances, threshold=0.1):
    """
    Validate the label transfer based on distance threshold.
    
    Parameters:
    distances: np.ndarrayneighborse label transfer
        
    Returns:
    np.ndarray: Boolean mask indicating reliable labels
    """
    return distances <= threshold

def visualize_results(source_pcd, target_pcd, transferred_labels, transformation_matrix):
    """
    Visualize the source and target point clouds with transferred labels.
    
    Parameters:
    source_pcd: o3d.geometry.PointCloud - Source point cloud
    target_pcd: o3d.geometry.PointCloud - Target point cloud
    transferred_labels: np.ndarray - Array of transferred labels
    transformation_matrix: np.ndarray - 4x4 transformation matrix
    """
    import open3d as o3d
    import matplotlib as mpl
    
    # Create color map excluding -1 labels
    valid_labels = transferred_labels[transferred_labels != -1]
    cmap = mpl.cm.tab20
    norm = mpl.colors.Normalize(vmin=min(valid_labels), vmax=max(valid_labels))
    
    # Create colors array
    colors = np.zeros((len(transferred_labels), 3))
    valid_mask = transferred_labels != -1
    colors[valid_mask] = cmap(norm(transferred_labels[valid_mask]))[:, :3]
    # Set invalid points to gray
    colors[~valid_mask] = [0.7, 0.7, 0.7]
    
    target_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    transformation_matrix[2, 3] += 2  # Offset for better visualization
    o3d.visualization.draw_geometries([source_pcd.transform(transformation_matrix), target_pcd])
