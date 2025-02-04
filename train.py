import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

# Vérifier si TensorFlow utilise le GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Paths
image_dir = "vertebrae-yolo-dataset/train/images"
ground_truth_dir = "vertebrae-yolo-dataset/train/ground_truth"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"

# Load dataset
def load_dataset(image_dir, ground_truth_dir):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(ground_truth_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print("letype des images", type(image), type(mask))
            images.append(image)
            masks.append(mask)
    return np.array(images), np.array(masks)

# Define U-Net model
def unet_model(input_size=(1024, 1024, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Prepare dataset
    images, masks = load_dataset(image_dir, ground_truth_dir)
    images = images / 255.0  # Normalize images
    masks = masks / 255.0  # Normalize masks

    # Create and train the model
    model = unet_model()

    # ----------------------------
    #   CHECKPOINT MANAGEMENT
    # ----------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt.weights.h5")

    ckpt = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=5)

    # Best model checkpoint
    best_ckpt = tf.train.Checkpoint(model=model)
    best_manager = tf.train.CheckpointManager(best_ckpt, BEST_MODEL_DIR, max_to_keep=1)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"[INFO] Restored from {manager.latest_checkpoint}")
    else:
        print("[INFO] Initializing from scratch...")

    # Define callbacks for checkpointing
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(BEST_MODEL_DIR, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            monitor='val_loss',
            mode='min'
        )
    ]

    model.fit(images,masks,batch_size=2, epochs=50,callbacks=callbacks)

    # Save the final model
    model.save('vertebrae_heart_segmentation_model.h5')

if __name__ == "__main__":
    main()