import os

import click
import numpy as np
import torch
import torchvision.transforms.functional
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from datasets.image_list_dataset import ImageListDataset
from utils.alignment import crop_faces
from utils.data_utils import get_expression_files, make_dataset_recursive_with_samples, create_subset, \
    select_random_images_per_identity
from utils.edit_utils import add_texts_to_image_vertical
from utils.image_utils import concat_images_horizontally
from utils.log_utils import save_latent_data, load_latent_data
from utils.models_utils import load_old_G, initialize_e4e_wplus

VIS_LOG_FREQUENCY = 100


def calc_mask(inversion, segmentation_model):
    background_classes = [0, 18, 16]
    inversion_resized = torch.cat([F.interpolate(inversion, (512, 512), mode='nearest')])
    inversion_normalized = transforms.functional.normalize(inversion_resized.clip(-1, 1).add(1).div(2),
                                                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    segmentation = segmentation_model(inversion_normalized)[0].argmax(dim=1, keepdim=True)
    is_foreground = torch.stack([segmentation != cls for cls in background_classes], dim=0).all(dim=0)
    foreground_mask = F.interpolate(is_foreground.float(), (1024, 1024), mode='bilinear', align_corners=True)
    return foreground_mask


def to_pil_image(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(to_np_image(tensor))


def to_np_image(tensor: torch.Tensor) -> Image.Image:
    x = (tensor[0].permute(1, 2, 0) + 1) * 255 / 2
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return x


def compute_mean_latent(expression_ds, e4e_inversion_net, e4e_image_transform, device='cuda', desc=None):
    """
    Compute the mean latent vector for a given dataset of images.

    Args:
        expression_ds (iterable): An iterable of (filename, image) pairs.
        e4e_inversion_net (nn.Module): The e4e inversion network.
        e4e_image_transform (callable): The image transform function for preprocessing.
        device (str): The device to run the computations on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A dictionary mapping filenames to latent vectors, and the mean latent vector.
    """
    ws = {}
    for fname, image in tqdm(expression_ds, desc=desc):
        w_pivot = get_e4e_inversion(image, e4e_inversion_net, e4e_image_transform, device)
        ws[fname] = w_pivot

    mean_w = torch.mean(torch.stack(list(ws.values())), dim=0)
    return ws, mean_w


def get_e4e_inversion(image, e4e_inversion_net, e4e_image_transform, device):
    new_image = e4e_image_transform(image).to(device)
    _, w = e4e_inversion_net(new_image.unsqueeze(0),
                             randomize_noise=False,
                             return_latents=True,
                             resize=False,
                             input_code=False)

    return w


@click.command()
@click.option('-s', '--starting_expression', type=str, help='Starting expression', required=True)
@click.option('-t', '--target_expression', type=str, help='Target expression to move towards', required=True)
@click.option('-d', '--dataset_root', type=str, required=True)
@click.option('--samples_list_dir', type=str, required=False)
@click.option('-l', '--latent_directory_root', type=str, required=True)
@click.option('-o', '--visualization_folder', type=str, help='Path to visualization folder', required=True)
@click.option('-n', '--num_frames_per_identity', type=int, help='Number of frames to use per ID', default=10)
@click.option('--scale', type=float, default=1.0, required=False)
@click.option('--xy_sigma', type=float, default=3.0)
@click.option('--center_sigma', type=float, default=1.0)
@click.option('--use_fa/--use_dlib', default=False, type=bool)
def main(**config):
    _main(**config, config=config)


def _main(starting_expression, target_expression, dataset_root, scale, center_sigma, xy_sigma, use_fa,
          visualization_folder, latent_directory_root, num_frames_per_identity, samples_list_dir, config):

    # latent_directory_root = os.path.join(latent_directory_root, f"num_frames_{num_frames_per_identity}")
    # loaded_starting_ws, loaded_starting_mean_w, \
    #     loaded_target_ws, loaded_target_mean_w, loaded_direction = load_latent_data(
    #     latent_directory_root, starting_expression, target_expression
    # )
    # print("Starting Mean W:", loaded_starting_mean_w.shape)
    # print("Direction:", loaded_direction.shape)
    #

    all_samples = make_dataset_recursive_with_samples(dataset_root, cache_file=samples_list_dir)
    print(f"Total data samples: {len(all_samples)}")

    # Filter samples for starting and target expressions
    starting_expression_files = get_expression_files(
        all_samples, starting_expression, num_frames=num_frames_per_identity, random_select=True
    )

    target_expression_files = get_expression_files(
        all_samples, target_expression, num_frames=num_frames_per_identity, random_select=True
    )

    print(f"Number of {starting_expression} samples: {len(starting_expression_files)}")
    print(f"Number of {target_expression} samples: {len(target_expression_files)}")

    image_size = 1024
    print('Aligning images')
    starting_crops, starting_orig_images, starting_quads, starting_filenames = crop_faces(image_size,
                                                                                          starting_expression_files,
                                                                                          scale,
                                                                                          center_sigma=center_sigma,
                                                                                          xy_sigma=xy_sigma,
                                                                                          use_fa=use_fa)

    target_crops, target_orig_images, target_quads, target_filenames = crop_faces(image_size,
                                                                                  target_expression_files,
                                                                                  scale,
                                                                                  center_sigma=center_sigma,
                                                                                  xy_sigma=xy_sigma,
                                                                                  use_fa=use_fa)

    print('Aligning completed')

    start_expression_ds = ImageListDataset(starting_crops,
                                           names=starting_filenames,
                                           source_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    target_expression_ds = ImageListDataset(target_crops,
                                            names=target_filenames,
                                            source_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    vis_dataset = select_random_images_per_identity(start_expression_ds)

    e4e_inversion_net = initialize_e4e_wplus().to('cuda')
    e4e_image_transform = transforms.Resize((256, 256))

    starting_ws, starting_mean_w = compute_mean_latent(
        start_expression_ds, e4e_inversion_net, e4e_image_transform, device='cuda', desc='Computing starting latent'
    )

    target_ws, target_mean_w = compute_mean_latent(
        target_expression_ds, e4e_inversion_net, e4e_image_transform, device='cuda', desc='Computing target latent'
    )

    direction = target_mean_w - starting_mean_w
    direction = direction / torch.linalg.norm(direction)

    latent_directory_root = os.path.join(latent_directory_root,
                                         f"{starting_expression}_to_{target_expression}",
                                         f"num_frames_{num_frames_per_identity}")
    os.makedirs(latent_directory_root, exist_ok=True)
    save_latent_data(latent_directory_root, starting_expression, target_expression,
                     starting_ws, starting_mean_w,
                     target_ws, target_mean_w,
                     direction)

    gen = load_old_G()

    # visualize the inversions. We wanna concat [[black, starting image, starting inversion, black],
    # [modified inversion x 1, modified inversion x 2, modified inversion x 3, modified inversion x 4]]
    visualization_folder = os.path.join(visualization_folder,
                                        f"{starting_expression}_to_{target_expression}",
                                        f"num_frames_{num_frames_per_identity}")
    os.makedirs(visualization_folder, exist_ok=True)
    expression_range = range(8)
    direction_scalar = 1

    with torch.no_grad():
        for fname, source_image in vis_dataset:
            w = starting_ws[fname]
            images_to_concat = []
            scales = []

            for range_multiplier in expression_range:
                scale = direction_scalar * range_multiplier
                delta_w = scale * direction
                inversion = gen.synthesis(w + delta_w, noise_mode='const', force_fp32=True)
                inversion = to_pil_image(inversion)
                scales.append(scale)
                images_to_concat.append(inversion)

            concat_images = concat_images_horizontally(*images_to_concat)
            images_with_text = add_texts_to_image_vertical([f'delta w = {scale}' for scale in scales],
                                                           concat_images)
            savefile_name = os.path.join(visualization_folder, fname.split('/')[-1])
            save_image(images_with_text, savefile_name)


def save_image(image, file):
    image = image.convert('RGB')
    image.save(file, quality=95)


if __name__ == '__main__':
    main()
