import numpy as np
import trimesh
import open3d as o3d

def get_semantic_bounding_boxes(mesh: trimesh.Trimesh, semantic_indices: list): # TODO bug for visual mesh
    """
    Get bounding boxes for each semantic part of the mesh.
    
    Args:
        mesh: trimesh.Trimesh object containing the full mesh
        semantic_indices: list of semantic indices for each vertex
        
    Returns:
        dict: Dictionary mapping semantic index to its bounding box (min_bound, max_bound)
    """
    vertices = mesh.vertices
    faces = mesh.faces
    unique_semantics = np.unique(semantic_indices)
    semantic_bboxes = {}
    
    # For each semantic label
    for semantic_idx in unique_semantics:
        # Get vertices that belong to this semantic part
        vertex_mask = np.array(semantic_indices) == semantic_idx
        semantic_vertices = vertices[vertex_mask]
        
        if len(semantic_vertices) > 0:
            # Calculate bounding box
            min_bound = np.min(semantic_vertices, axis=0)
            max_bound = np.max(semantic_vertices, axis=0)
            semantic_bboxes[int(semantic_idx)] = (min_bound, max_bound)
    
    return semantic_bboxes

def visualize_semantic_bboxes(mesh, semantic_bboxes):
    """
    Visualize mesh with semantic bounding boxes.
    
    Args:
        mesh: o3d.geometry.TriangleMesh
        semantic_bboxes: dict mapping semantic index to (min_bound, max_bound)
    """
    geometries = [mesh]
    
    # Create lines for each bounding box with different colors
    colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]  # Add more colors if needed
    
    for idx, (semantic_idx, (min_bound, max_bound)) in enumerate(semantic_bboxes.items()):
        # Create bounding box points
        points = [
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]]
        ]
        
        # Define lines connecting box vertices
        lines = [
            [0,1], [1,2], [2,3], [3,0],  # Bottom face
            [4,5], [5,6], [6,7], [7,4],  # Top face
            [0,4], [1,5], [2,6], [3,7]   # Connecting edges
        ]
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set color
        color = colors[idx % len(colors)]
        line_set.paint_uniform_color(color)
        
        geometries.append(line_set)
    
    # Visualize
    o3d.visualization.draw_geometries(geometries)
