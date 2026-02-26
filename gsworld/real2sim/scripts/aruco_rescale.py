#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# pycolmap = 0.4.0
from gsworld.real2sim.aruco_estimator.aruco_scale_factor import ArucoScaleFactor
from colmap_wrapper.colmap import COLMAP

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Process COLMAP project with Aruco marker scaling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--source_path',
        type=str,
        help='Path to the COLMAP project folder'
    )
    
    parser.add_argument(
        '--aruco-size',
        type=float,
        default=0.100,
        help='Size of the aruco marker in meters'
    )
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Validate source path
    source_path = Path(args.source_path)
    if not source_path.exists():
        print(f"Error: Source path '{source_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if not source_path.is_dir():
        print(f"Error: Source path '{source_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    
    aruco_size = args.aruco_size
    
    print(f"Processing COLMAP project: {source_path}")
    print(f"Aruco marker size: {aruco_size} meters")
    
    # Load Colmap project folder
    project = COLMAP(project_path=str(source_path))

    # Init & run pose estimation of corners in 3D & estimate mean L2 distance between the four aruco corners
    aruco_scale_factor = ArucoScaleFactor(photogrammetry_software=project, aruco_size=aruco_size)
    aruco_distance, aruco_corners_3d = aruco_scale_factor.run()
    print('Size of the unscaled aruco markers: ', aruco_distance)

    # Calculate scaling factor, apply it to the scene and save scaled point cloud
    dense, scale_factor = aruco_scale_factor.apply() 
    print('Point cloud and poses are scaled by: ', scale_factor)
    print('Size of the scaled (true to scale) aruco markers in meters: ', aruco_distance * scale_factor)

    # Write Data
    aruco_scale_factor.write_data()
    
    print("Processing completed successfully!")

if __name__ == "__main__":
    main()