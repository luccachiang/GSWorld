import json
import h5py
import numpy as np
import os
import imageio_ffmpeg as ffmpeg
import subprocess
import imageio
import tqdm
from typing import List, Optional
from PIL import Image

def hdf5_serialization(data, path='data.h5', verbose=True):
    with h5py.File(path, 'w') as hf:
        def _save_to_hdf5(d, group):
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    group.create_dataset(k, data=v, compression='gzip')
                elif isinstance(v, dict):
                    subgroup = group.create_group(k)
                    _save_to_hdf5(v, subgroup)
                else:
                    group.attrs[k] = v
        
        _save_to_hdf5(data, hf)

def read_hdf5_to_dict_recursively(filepath, verbose=True):
    def load_group_or_dataset(obj):
        """
        Recursively convert HDF5 groups and datasets to dictionary
        """
        if isinstance(obj, h5py.Dataset):
            return obj[:]
        elif isinstance(obj, h5py.Group):
            return {k: load_group_or_dataset(obj[k]) for k in obj.keys()}
        elif isinstance(obj, h5py.AttributeManager):
            return dict(obj)
    
    with h5py.File(filepath, 'r') as f:
        # Convert entire file to a dictionary
        data_dict = load_group_or_dataset(f) # ndarray
        
        # Pretty print the converted dictionary
        if verbose:
            import pprint
            pprint.pprint(data_dict)
        
        return data_dict
    
def save_images_to_mp4(hdf5_path, output_dir, save_frame=False, fps=30):
    with h5py.File(hdf5_path, 'r') as f:
        for traj in sorted(f.keys(), key=lambda x: int(x.split('_')[-1])):
            for key in f[traj]['obs']['sensor_data'].keys():
                images = f[traj]['obs']['sensor_data'][key]['rgb'][:]  # (N, H, W, C)

                if images.ndim == 4 and images.shape[-1] in [1, 3, 4]:  # (N, H, W, C)
                    height, width = images.shape[1:3]
                    traj_dir = os.path.join(output_dir, traj)
                    frame_dir = os.path.join(traj_dir, f"{key}_frames")
                    os.makedirs(frame_dir, exist_ok=True)
                    os.makedirs(traj_dir, exist_ok=True)
                    output_path = os.path.join(traj_dir, f"{key}.mp4")

                    # FFmpeg process setup
                    writer = ffmpeg.get_ffmpeg_exe()
                    process = subprocess.Popen(
                        [writer, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                         '-s', f'{width}x{height}', '-pix_fmt', 'rgb24',
                         '-r', str(fps), '-i', '-', '-an',
                         '-vcodec', 'libx264', output_path],
                        stdin=subprocess.PIPE
                    )

                    for idx, img in enumerate(images):
                        if img.shape[-1] == 1:  # Grayscale
                            img = np.repeat(img, 3, axis=-1)
                        elif img.shape[-1] == 4:  # RGBA
                            img = img[..., :3]

                        img_uint8 = img.astype(np.uint8)
                        
                        # Save frame as image
                        if save_frame:
                            Image.fromarray(img_uint8).save(os.path.join(frame_dir, f"frame_{idx:04d}.png"))

                        # Write to video
                        process.stdin.write(img_uint8.tobytes())

                    process.stdin.close()
                    process.wait()
                    print(f"Saved {output_path} and frames to {frame_dir}")
                else:
                    print(f"Skipping {key}: Unexpected shape {images.shape}")

def save_image_frames(
    images: List[np.ndarray],
    output_dir: str,
    prefix: str = "frame",
    verbose: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Saving {len(images)} images to {output_dir}")
        images_iter = tqdm.tqdm(enumerate(images))
    else:
        images_iter = enumerate(images)

    for i, im in images_iter:
        image_path = os.path.join(output_dir, f"{prefix}_{i:05d}.png")
        imageio.imwrite(image_path, im)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)