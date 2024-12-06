import os
from collections import defaultdict
from dataclasses import dataclass
import random
from typing import List

import torch
from PIL import Image
from torch.utils.data import Subset

from datasets.image_list_dataset import ImageListDataset


@dataclass
class DataSample:
    file_path: str
    identity: str
    pose: str
    expression: str
    level: str
    frame: str


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in sorted(os.listdir(dir)):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            fname = fname.split('.')[0]
            images.append((fname, path))
    return images


def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset_recursive(root_dir):
    """
    Recursively collect all image filenames and paths in directories containing 'frames'.
    Returns a list of tuples: (filename, path).
    """
    images = []
    assert os.path.isdir(root_dir), f"{root_dir} is not a valid directory"
    for root, dirs, fnames in os.walk(root_dir):
        # Skip directories that do not contain 'frames'
        if "frames" not in root:
            continue
        for fname in fnames:
            if fname.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                path = os.path.join(root, fname)
                name = os.path.splitext(fname)[0]  # Get filename without extension
                images.append((name, path))
    return images


def make_dataset_recursive_with_samples(root_dir) -> List[DataSample]:
    """
    Recursively collect all images in directories containing 'frames' and
    create DataSample instances with metadata.
    """
    samples = []
    assert os.path.isdir(root_dir), f"{root_dir} is not a valid directory"

    for root, dirs, fnames in os.walk(root_dir):
        # Skip non-relevant directories
        if "frames" not in root:
            continue
        for fname in fnames:
            if fname.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, fname)
                # Extract metadata from the file path
                parts = file_path.split(os.sep)
                identity = parts[-7]
                pose = parts[-6]
                expression = parts[-5]
                level = parts[-4]
                frame = os.path.splitext(fname)[0]
                samples.append(DataSample(file_path, identity, pose, expression, level, frame))
    return samples


def get_expression_files(samples: List[DataSample], expression: str, num_frames: int = 10,
                         random_select: bool = True) -> List[tuple]:
    """
    Filter DataSample objects by expression and select a limited number of frames per video.
    Return (fname, filepath) pairs for the selected frames.

    Args:
        samples (List[DataSample]): List of all DataSample objects.
        expression (str): Expression to filter (e.g., "neutral").
        num_frames (int): Number of frames to select per video.
        random_select (bool): If True, select frames randomly. Otherwise, select the first and last frames.

    Returns:
        List[tuple]: List of (fname, filepath) pairs.
    """
    filtered_samples = [sample for sample in samples if sample.expression == expression]

    # Group samples by identity and pose (assume each combination corresponds to a video)
    grouped_samples = {}
    for sample in filtered_samples:
        key = (sample.identity, sample.pose, sample.expression, sample.level)
        if key not in grouped_samples:
            grouped_samples[key] = []
        grouped_samples[key].append(sample)

    # Select frames for each group
    selected_pairs = []
    for group, frames in grouped_samples.items():
        if random_select:
            selected = random.sample(frames, min(num_frames, len(frames)))
        else:
            selected = frames[:num_frames] + frames[-num_frames:]

        # Convert DataSample objects to (fname, filepath) pairs
        selected_pairs.extend([(sample.frame, sample.file_path) for sample in selected])

    return selected_pairs


def create_subset(dataset, subset_size, seed=None):
    """
    Create a random subset of a PyTorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        subset_size (int): Number of samples to include in the subset.
        seed (int, optional): Seed for reproducibility. Default is None.

    Returns:
        torch.utils.data.Subset: A subset of the original dataset.
    """
    if subset_size > len(dataset):
        raise ValueError(f"Subset size {subset_size} cannot be larger than dataset size {len(dataset)}.")

    # Set the random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random indices
    indices = torch.randperm(len(dataset))[:subset_size].tolist()

    # Create and return the subset
    return Subset(dataset, indices)


def select_random_images_per_identity(source_dataset):
    """
    Select one random image per identity from the dataset.

    Args:
        dataset (ImageListDataset): The original dataset.
        data_samples (List[DataSample]): List of DataSample objects containing metadata.

    Returns:
        ImageListDataset: A new dataset with one random image per identity.
    """
    # Group samples by identity
    identity_to_samples = defaultdict(list)
    for item in source_dataset:
        file_path = item[0]
        parts = file_path.split(os.sep)
        identity = parts[-7]
        identity_to_samples[identity].append(file_path)

    # Randomly select one sample per identity
    selected_indices = []
    for identity, samples in identity_to_samples.items():
        random_sample = random.choice(samples)
        selected_indices.append(source_dataset.names.index(random_sample))

    # Create a new dataset with the selected indices
    selected_images = [source_dataset.images[i] for i in selected_indices]
    selected_names = [source_dataset.names[i] for i in selected_indices]

    return ImageListDataset(selected_images, source_dataset.source_transform, selected_names)
