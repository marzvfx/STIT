import os

import numpy as np
from PIL import Image
import wandb
from configs import global_config
import torch


def log_image_from_w(w, G, name):
    img = get_image_from_w(w, G)
    pillow_image = Image.fromarray(img)
    wandb.log(
        {f"{name}": [
            wandb.Image(pillow_image, caption=f"current inversion {name}")]}
    )


def log_images_from_w(ws, G, names):
    for name, w in zip(names, ws):
        w = w.to(global_config.device)
        log_image_from_w(w, G, name)


def save_image(name, method_type, results_dir, image, run_id):
    image.save(f'{results_dir}/{method_type}_{name}_{run_id}.jpg')


def save_w(w, G, name, method_type, results_dir):
    im = get_image_from_w(w, G)
    im = Image.fromarray(im, mode='RGB')
    save_image(name, method_type, results_dir, im)


def save_concat_image(base_dir, image_latents, new_inv_image_latent, new_G,
                      old_G,
                      file_name,
                      extra_image=None):
    images_to_save = []
    if extra_image is not None:
        images_to_save.append(extra_image)
    for latent in image_latents:
        images_to_save.append(get_image_from_w(latent, old_G))
    images_to_save.append(get_image_from_w(new_inv_image_latent, new_G))
    result_image = create_alongside_images(images_to_save)
    result_image.save(f'{base_dir}/{file_name}.jpg')


def save_single_image(base_dir, image_latent, G, file_name):
    image_to_save = get_image_from_w(image_latent, G)
    image_to_save = Image.fromarray(image_to_save, mode='RGB')
    image_to_save.save(f'{base_dir}/{file_name}.jpg')


def create_alongside_images(images):
    res = np.concatenate([np.array(image) for image in images], axis=1)
    return Image.fromarray(res, mode='RGB')


def get_image_from_w(w, G):
    if len(w.size()) <= 2:
        w = w.unsqueeze(0)
    with torch.no_grad():
        img = G.synthesis(w, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
    return img[0]


def save_latent_data(save_dir, starting_expression, target_expression, starting_ws, starting_mean_w, target_ws,
                     target_mean_w, direction):
    """
    Save latent data to a directory.

    Args:
        save_dir (str): Directory to save the data.
        starting_expression (str): Starting expression label.
        target_expression (str): Target expression label.
        starting_ws (dict): Dictionary of {filename: torch tensor}.
        starting_mean_w (torch.Tensor): Mean latent vector for the starting expression.
        target_ws (dict): Dictionary of {filename: torch tensor}.
        target_mean_w (torch.Tensor): Mean latent vector for the target expression.
        direction (torch.Tensor): Direction vector from starting to target.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_prefix = f"{starting_expression}_to_{target_expression}"

    # Save the individual components
    torch.save(starting_ws, os.path.join(save_dir, f"{file_prefix}_starting_ws.pt"))
    torch.save(starting_mean_w, os.path.join(save_dir, f"{file_prefix}_starting_mean_w.pt"))
    torch.save(target_ws, os.path.join(save_dir, f"{file_prefix}_target_ws.pt"))
    torch.save(target_mean_w, os.path.join(save_dir, f"{file_prefix}_target_mean_w.pt"))
    torch.save(direction, os.path.join(save_dir, f"{file_prefix}_direction.pt"))


def load_latent_data(save_dir, starting_expression, target_expression):
    """
    Load latent data from a directory.

    Args:
        save_dir (str): Directory to load the data from.
        starting_expression (str): Starting expression label.
        target_expression (str): Target expression label.

    Returns:
        tuple: (starting_ws, starting_mean_w, target_ws, target_mean_w, direction)
    """
    file_prefix = f"{starting_expression}_to_{target_expression}"

    starting_ws = torch.load(os.path.join(save_dir, f"{file_prefix}_starting_ws.pt"))
    starting_mean_w = torch.load(os.path.join(save_dir, f"{file_prefix}_starting_mean_w.pt"))
    target_ws = torch.load(os.path.join(save_dir, f"{file_prefix}_target_ws.pt"))
    target_mean_w = torch.load(os.path.join(save_dir, f"{file_prefix}_target_mean_w.pt"))
    direction = torch.load(os.path.join(save_dir, f"{file_prefix}_direction.pt"))

    return starting_ws, starting_mean_w, target_ws, target_mean_w, direction
