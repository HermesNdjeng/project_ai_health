import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms as T

from utils.logging_utils import logger  # Import the logger


def v_h_s(points):
    """Compute VHS score from 6 predicted keypoints (L, S, T axes)."""
    A, B, C, D, E, F = points.reshape(6, 2)
    AB = np.linalg.norm(A - B)
    CD = np.linalg.norm(C - D)
    EF = np.linalg.norm(E - F)
    VHS = 6 * (AB + CD) / EF
    logger.info(f"VHS calculation: L={AB:.2f}, S={CD:.2f}, T={EF:.2f}, VHS={VHS:.2f}")
    return VHS


def load_model(checkpoint_path):
    """
    Load the VHS prediction model from a checkpoint.
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    start_time = datetime.now()

    # Create base model
    model = models.efficientnet_b7(pretrained=False)

    # Modify the classifier to match the expected output size (12 values for 6 coordinate points)
    in_features = model.classifier[1].in_features  # Should be 2560 for EfficientNet-B7
    model.classifier[1] = nn.Linear(in_features, 12)

    # Now load the checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))  # type: ignore
    model.eval()

    load_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
    return model


def get_transform(resized_image_size):
    """Build the inference transform pipeline (resize, normalize)."""
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Resize(size=(resized_image_size, resized_image_size)))
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def visualize_prediction_measurements(
    image_path, model, transform, device, ax, resized_image_size=300
):
    """
    Visualize the prediction and extract L, S, T values and VHS score from a radiograph.

    This function is modified to also display individual L, S, T values.
    """
    logger.info(f"Processing image: {image_path}")
    start_time = datetime.now()

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    logger.info(f"Image dimensions: {w}x{h}")

    img_transformed = transform(img).unsqueeze(0).to(device)

    # Log CNN inference
    logger.info("Running CNN inference for key point detection")
    cnn_start_time = datetime.now()

    with torch.no_grad():
        pred_points = model(img_transformed).cpu().numpy().reshape(6, 2)
        pred_points = pred_points * resized_image_size
        pred_points[:, 0] = w / resized_image_size * pred_points[:, 0]
        pred_points[:, 1] = h / resized_image_size * pred_points[:, 1]

        # Extract the points
        A, B, C, D, E, F = pred_points

        # Calculate L, S, T values
        l_value = np.linalg.norm(A - B)  # Long axis (L)
        s_value = np.linalg.norm(C - D)  # Short axis (S)
        t_value = np.linalg.norm(E - F)  # Reference vertebral length (T)

        # Calculate VHS score
        vhs_score = 6 * (l_value + s_value) / t_value

    cnn_time = (datetime.now() - cnn_start_time).total_seconds()
    logger.info(f"CNN inference completed in {cnn_time:.2f} seconds")
    logger.info(
        f"Measurements - L: {l_value:.2f}, S: {s_value:.2f}, T: {t_value:.2f}, VHS: {vhs_score:.2f}"
    )

    # Display the image and points
    ax.imshow(img)
    ax.scatter(pred_points[:, 0], pred_points[:, 1], c="red", marker="o", label="Predicted")

    # Draw lines between point pairs
    for p1, p2 in [(0, 1), (2, 3), (4, 5)]:
        ax.plot(
            [pred_points[p1, 0], pred_points[p2, 0]],
            [pred_points[p1, 1], pred_points[p2, 1]],
            "r--",
        )

    # Display all values
    ax.set_title(
        f"{os.path.basename(image_path)}\n"
        f"L: {l_value:.2f}, S: {s_value:.2f}, T: {t_value:.2f}\n"
        f"VHS: {vhs_score:.2f}"
    )
    ax.axis("off")
    ax.set_aspect("equal")

    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Image processing completed in {total_time:.2f} seconds")

    return l_value, s_value, t_value, vhs_score
