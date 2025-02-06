import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

# Vérifier si TensorFlow utilise le GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Paths
image_dir = "vertebrae-yolo-dataset/train/images"
ground_truth_dir = "vertebrae-yolo-dataset/train/ground_truth"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"

# Dimensions des images (par exemple, 512x512)
img_width = 512
img_height = 512

# Générateur de données
class DataGenerator(Sequence):
    def __init__(self, image_dir, ground_truth_dir, batch_size=32, img_size=(512, 512), shuffle=True):
        self.image_dir = image_dir
        self.ground_truth_dir = ground_truth_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_filenames = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks = self.__data_generation(batch_filenames)
        if len(images) == 0 or len(masks) == 0:
            return self.__getitem__((index + 1) % self.__len__())
        return images, masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_filenames)

    def __data_generation(self, batch_filenames):
        images = []
        masks = []
        for filename in batch_filenames:
            image_path = os.path.join(self.image_dir, filename)
            mask_filename = filename.replace(".jpg", ".png").replace(".png", ".png")
            mask_path = os.path.join(self.ground_truth_dir, mask_filename)

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is not None and mask is not None:
                image = cv2.resize(image, self.img_size)
                mask = cv2.resize(mask, self.img_size)
                images.append(image / 255.0)
                masks.append(mask / 255.0)

        images = np.array(images)
        masks = np.expand_dims(np.array(masks), axis=-1)
        return images, masks

# Define U-Net model
def unet_model(input_size=(512, 512, 3)):
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
    # Create data generators
    train_generator = DataGenerator(image_dir, ground_truth_dir, batch_size=1, img_size=(img_width, img_height))
    val_generator = DataGenerator(image_dir, ground_truth_dir, batch_size=1, img_size=(img_width, img_height), shuffle=False)

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
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    model.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=callbacks)

    # Save the final model
    model.save('vertebrae_heart_segmentation_model.h5')

if __name__ == "__main__":
    main()