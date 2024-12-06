import cv2
import numpy as np
import torch
from PIL import Image


def concat_images_horizontally(*images: Image.Image):
    assert all(image.height == images[0].height for image in images)
    total_width = sum(image.width for image in images)

    new_im = Image.new(images[0].mode, (total_width, images[0].height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    return new_im


def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.squeeze(0).permute(1, 2, 0).add(1).mul(255).div(2).squeeze()
    x = x.detach().cpu().numpy()
    x = np.rint(x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def add_text_to_image(image: np.ndarray, text: str, position: tuple = (50, 50),
                      font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                      color=(255, 255, 255), thickness=2) -> np.ndarray:
    """
    Add text to an image represented as a NumPy array.

    Args:
        image (np.ndarray): The input image array (H x W x C).
        text (str): The text to add to the image.
        position (tuple): The (x, y) position to place the text.
        font (int): The font type (e.g., cv2.FONT_HERSHEY_SIMPLEX).
        font_scale (float): Font scale (size).
        color (tuple): Color of the text in BGR format.
        thickness (int): Thickness of the text stroke.

    Returns:
        np.ndarray: The image with the added text.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] not in [3, 4]:
        raise ValueError("Input image must be a color image (H x W x C).")

    # Add text to the image
    annotated_image = image.copy()
    cv2.putText(annotated_image, text, position, font, font_scale, color, thickness)
    return annotated_image
