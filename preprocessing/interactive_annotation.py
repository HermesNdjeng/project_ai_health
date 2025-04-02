import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import os

# Load SAM model
MODEL_TYPE = "vit_l"  # Choose model type: vit_b, vit_l, vit_h
CHECKPOINT_PATH = "C:/Users/nzooh/OneDrive - aivancity/Desktop/Deployement exercise/Project_root/models/sam_vit_l_0b3195.pth"  # Path to SAM model checkpoint

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)

# Directories for input and output
IMAGE_DIR = "filtered_data_png/"  # Directory with input images
OUTPUT_DIR = "labels/"  # Directory for YOLO annotations
os.makedirs(OUTPUT_DIR, exist_ok=True)

def click_event(event, x, y, flags, params):
    """Callback function for mouse click."""
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click to select a point
        clicks.append((x, y))
        print(f"Added point: {x, y}")
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to finish annotation
        print("Finished annotating object.")
        cv2.destroyAllWindows()

def mask_to_bbox(mask):
    """Convert a binary mask to bounding box."""
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return x_min, y_min, x_max, y_max

def normalize_bbox(x_min, y_min, x_max, y_max, image_width, image_height):
    """Normalize bounding box for YOLO."""
    center_x = (x_min + x_max) / 2 / image_width
    center_y = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return center_x, center_y, width, height

# Interactive segmentation
for image_name in os.listdir(IMAGE_DIR):
    if not image_name.endswith((".jpg", ".png")):
        continue

    # Load image
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Set up SAM with the image
    predictor.set_image(image)

    # Interactive prompt selection
    print(f"Annotating {image_name}.")
    print("Left-click to add points, right-click to finish.")
    clicks = []

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)  # Wait until right-click closes the window

    # Generate segmentation mask from points
    if len(clicks) > 0:
        input_points = np.array(clicks)
        input_labels = np.ones(len(input_points))  # Labels: 1 for foreground
        masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)

        # Use the first mask (modify if needed for multiple objects)
        yolo_annotations = []
        for mask in masks:
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue

            x_min, y_min, x_max, y_max = bbox
            yolo_bbox = normalize_bbox(x_min, y_min, x_max, y_max, width, height)

            # Assuming class ID 0 for all objects; adjust as needed
            class_id = 0
            yolo_annotations.append(f"{class_id} " + " ".join(map(str, yolo_bbox)))

            # Optional: visualize the mask
            segmented_image = image.copy()
            segmented_image[mask] = [0, 255, 0]  # Highlight in green
            cv2.imshow("Segmented", segmented_image)
            cv2.waitKey(500)

        # Save YOLO annotations
        annotation_file = os.path.join(OUTPUT_DIR, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(annotation_file, "w") as f:
            f.write("\n".join(yolo_annotations))

        print(f"Annotations saved for {image_name}.")

cv2.destroyAllWindows()
