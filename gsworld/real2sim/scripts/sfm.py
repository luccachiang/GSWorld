#!/usr/bin/env python3

import os
import logging
from argparse import ArgumentParser
import subprocess

def run_cmd(cmd, name):
    logging.info(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        logging.error(f"{name} failed")
        logging.error(result.stderr)
        raise RuntimeError(f"{name} failed")

def main():
    parser = ArgumentParser(description="Run COLMAP feature extraction, matching, and bundle adjustment")
    
    # Required arguments
    parser.add_argument("--source_path", 
                          required=True,
                       help="Path to the source directory containing images")
    
    # Optional arguments
    parser.add_argument("--colmap-command", 
                       default="colmap",
                       help="COLMAP command/executable path (default: colmap)")
    
    parser.add_argument("--camera", 
                       default="PINHOLE",
                       choices=["PINHOLE", "SIMPLE_PINHOLE", "RADIAL", "OPENCV", "SIMPLE_RADIAL"],
                       help="Camera model (default: PINHOLE)")
    
    parser.add_argument("--no-gpu", 
                       action="store_true",
                       help="Disable GPU usage")
    
    parser.add_argument("--skip-matching", 
                       action="store_true",
                       help="Skip the matching process")
    
    parser.add_argument("--keep-distorted", 
                       action="store_true",
                       help="Keep the distorted folder after processing")
    
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Extract arguments
    colmap_command = args.colmap_command
    source_path = os.path.abspath(args.source_path)
    camera = args.camera
    no_gpu = args.no_gpu
    skip_matching = args.skip_matching
    keep_distorted = args.keep_distorted
    
    # Validate source path
    if not os.path.exists(source_path):
        logging.error(f"Source path does not exist: {source_path}")
        exit(1)
    
    images_path = os.path.join(source_path, "images")
    if not os.path.exists(images_path):
        logging.error(f"Images directory does not exist: {images_path}")
        exit(1)
    
    use_gpu = 0 if no_gpu else 1
    
    if args.verbose:
        logging.info(f"Source path: {source_path}")
        logging.info(f"Camera model: {camera}")
        logging.info(f"Using GPU: {not no_gpu}")
        logging.info(f"Skip matching: {skip_matching}")
    
    if not skip_matching:
        distorted_path = os.path.join(source_path, "distorted")
        sparse_distorted_path = os.path.join(distorted_path, "sparse")
        os.makedirs(sparse_distorted_path, exist_ok=True)
        
        database_path = os.path.join(distorted_path, "database.db")
        
        ## Feature extraction
        logging.info("Starting feature extraction...")
        if os.path.exists(database_path):
            logging.info("Removing existing database.db")
            os.remove(database_path)
        run_cmd(f"{colmap_command} feature_extractor " \
            f"--database_path {database_path} " \
            f"--image_path {images_path} " \
            f"--ImageReader.single_camera 1 " \
            f"--ImageReader.camera_model {camera} " \
            f"--SiftExtraction.use_gpu {use_gpu}", "Feature extraction")
        
        ## Feature matching
        logging.info("Starting feature matching...")
        run_cmd(f"{colmap_command} exhaustive_matcher " \
            f"--database_path {database_path} " \
            f"--SiftMatching.use_gpu {use_gpu} " , "Feature matching")
        
        ### Bundle adjustment
        logging.info("Starting bundle adjustment...")
        # The default Mapper tolerance is unnecessarily large,
        # decreasing it speeds up bundle adjustment steps.
        run_cmd(f"{colmap_command} mapper " \
            f"--database_path {database_path} " \
            f"--image_path {images_path} " \
            f"--output_path {sparse_distorted_path} " \
            f"--Mapper.ba_global_function_tolerance=0.000001", "Bundle adjustment")
        
        ### Convert the model to a text file
        logging.info("Converting model to text format...")
        sparse_output_path = os.path.join(source_path, "sparse")
        os.makedirs(sparse_output_path, exist_ok=True)
        
        sparse_input_path = os.path.join(sparse_distorted_path, "0")
        run_cmd(f"{colmap_command} model_converter " \
            f"--input_path {sparse_input_path} " \
            f"--output_path {sparse_output_path} " \
            f"--output_type TXT", "Model conversion")
        
        ### Delete the distorted folder (unless --keep-distorted is specified)
        if not keep_distorted:
            logging.info("Cleaning up distorted folder...")
            cleanup_cmd = f"rm -r {distorted_path}"
            if args.verbose:
                logging.info(f"Running: {cleanup_cmd}")
            os.system(cleanup_cmd)
            logging.info("Cleanup completed.")
        else:
            logging.info("Keeping distorted folder as requested.")
            
        # TODO delete some files in the sparse folder
        os.system(f"rm -r {sparse_output_path}/frames.txt")
        os.system(f"rm -r {sparse_output_path}/rigs.txt")
    
    logging.info("COLMAP processing completed successfully!")

if __name__ == "__main__":
    main()