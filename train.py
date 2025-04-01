import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import argparse
import shutil

# Vérifier si TensorFlow utilise le GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Paths
train_image_dir = "vertebrae-yolo-dataset/train/images"
valid_image_dir = "vertebrae-yolo-dataset/valid/images"
train_ground_truth_dir = "vertebrae-yolo-dataset/train/ground_truth_annotated"
valid_ground_truth_dir = "vertebrae-yolo-dataset/valid/ground_truth_annotated"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"

# Dimensions des images
img_width = 512
img_height = 512
num_classes = 3  # Fond, cœur, vertèbres

# Fonction de perte Dice (plus stable pour la segmentation)
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Poids inverses à la fréquence des classes
class_weights = [0.1, 0.45, 0.45]  # Exemple: fond=0.1, cœur=0.45, vertèbres=0.45

# Fonction de perte pondérée
def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

# Générateur de données
class DataGenerator(Sequence):
    def __init__(self, image_dir, ground_truth_dir, batch_size=32, img_size=(512, 512), num_classes=3, shuffle=True):
        self.image_dir = image_dir
        self.ground_truth_dir = ground_truth_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        
        # Vérifier que les dossiers existent
        if not os.path.exists(image_dir):
            raise ValueError(f"Le dossier d'images n'existe pas: {image_dir}")
        if not os.path.exists(ground_truth_dir):
            raise ValueError(f"Le dossier d'annotations n'existe pas: {ground_truth_dir}")
            
        # Obtenir uniquement les fichiers qui ont à la fois une image et un masque
        all_images = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
        self.image_filenames = []
        
        for img in all_images:
            mask_path = os.path.join(ground_truth_dir, img.replace(".jpg", ".png").replace(".png", ".png"))
            if os.path.exists(mask_path):
                self.image_filenames.append(img)
                
        print(f"Trouvé {len(self.image_filenames)} images avec masques correspondants dans {image_dir}")
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.floor(len(self.image_filenames) / self.batch_size)))

    def __getitem__(self, index):
        batch_filenames = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks = self.__data_generation(batch_filenames)
        if len(images) == 0 or len(masks) == 0:
            # Retourner des tableaux vides plutôt que d'appeler récursivement
            return np.empty((0, *self.img_size, 3)), np.empty((0, *self.img_size, self.num_classes))
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

                # Remapper les valeurs du masque pour les classes correctes
                # 0 -> 0 (fond)
                # 29 -> 1 (rouge converti en gris ~29)
                # 149 -> 2 (vert converti en gris ~149)
                remapped_mask = np.zeros_like(mask)
                remapped_mask[mask == 0] = 0
                remapped_mask[mask == 29] = 1
                remapped_mask[mask == 149] = 2
                
                # Vérifier que les valeurs de masque sont dans la plage attendue
                unique_values = np.unique(remapped_mask)
                if np.max(unique_values) >= self.num_classes:
                    print(f"Attention: Masque {mask_path} contient encore des valeurs inattendues après remappage: {unique_values}")
                
                # Convertir le masque en one-hot encoding
                mask_one_hot = np.zeros((*self.img_size, self.num_classes), dtype=np.float32)
                for c in range(self.num_classes):
                    mask_one_hot[:, :, c] = (remapped_mask == c).astype(np.float32)

                images.append(image / 255.0)
                masks.append(mask_one_hot)

        images = np.array(images, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        return images, masks

# Attention Gate
def attention_gate(x, g, filters):
    """Attention Gate module to focus on relevant features.
    x: Skip connection features from encoder
    g: Features from decoder
    filters: Number of filters
    """
    # Theta_x for skip connection (encoder features)
    theta_x = Conv2D(filters, 1, use_bias=True)(x)
    
    # Phi_g for gating signal from decoder
    phi_g = Conv2D(filters, 1, use_bias=True)(g)
    
    # Combine signals (compatibility score)
    f = tf.keras.layers.Activation('relu')(theta_x + phi_g)
    
    # Attention coefficient
    psi_f = Conv2D(1, 1, use_bias=True)(f)
    att_map = tf.keras.layers.Activation('sigmoid')(psi_f)
    
    # Attend to features
    y = x * att_map
    
    return y

# Architecture U-Net corrigée
def unet_model(input_size=(512, 512, 3), num_classes=3):
    inputs = Input(input_size)
    
    # Encoder
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

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder - Correction des bugs ici
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    # Correction: utiliser up8 au lieu de conv8 non défini
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)  
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Couche de sortie
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    # Utilisation d'un taux d'apprentissage plus faible et clipping de gradient
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    
    # Compilation avec métriques supplémentaires
    model.compile(
        optimizer=optimizer, 
        loss=weighted_categorical_crossentropy(class_weights),
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=num_classes)]
    )
    
    return model

# Attention U-Net model
def attention_unet_model(input_size=(512, 512, 3), num_classes=3):
    """
    Implementation of Attention U-Net for medical image segmentation
    """
    inputs = Input(input_size)
    
    # Encoder
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

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder with attention gates
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    # Apply attention gate to skip connection
    att6 = attention_gate(conv4, up6, 512)
    merge6 = concatenate([up6, att6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    # Apply attention gate to skip connection
    att7 = attention_gate(conv3, up7, 256)
    merge7 = concatenate([up7, att7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    # Apply attention gate to skip connection
    att8 = attention_gate(conv2, up8, 128)
    merge8 = concatenate([up8, att8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    # Apply attention gate to skip connection
    att9 = attention_gate(conv1, up9, 64)
    merge9 = concatenate([up9, att9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output layer
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    # Use a lower learning rate and gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    
    # Fixed weighted_categorical_crossentropy to avoid TF variable creation issues
    def weighted_categorical_crossentropy_fixed(weights_list):
        """Version compatible with tf.function"""
        weights_tensor = tf.constant(weights_list)
        
        def loss(y_true, y_pred):
            y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
            y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
            loss = y_true * tf.math.log(y_pred) * weights_tensor
            loss = -tf.reduce_sum(loss, -1)
            return loss
        
        return loss
    
    # Compilation with metrics
    model.compile(
        optimizer=optimizer, 
        loss=weighted_categorical_crossentropy_fixed(class_weights),
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=num_classes)]
    )
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train U-Net model for segmentation")
    parser.add_argument('--start-from-scratch', action='store_true', help="Start training from scratch and delete previous checkpoints")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training")
    args = parser.parse_args()

    if args.start_from_scratch:
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
        if os.path.exists(BEST_MODEL_DIR):
            shutil.rmtree(BEST_MODEL_DIR)
        print("[INFO] Starting from scratch and deleting previous checkpoints...")

    # Création des générateurs de données
    train_generator = DataGenerator(
        train_image_dir, 
        train_ground_truth_dir, 
        batch_size=args.batch_size, 
        img_size=(img_width, img_height), 
        num_classes=num_classes
    )
    
    val_generator = DataGenerator(
        valid_image_dir, 
        valid_ground_truth_dir, 
        batch_size=args.batch_size, 
        img_size=(img_width, img_height), 
        num_classes=num_classes, 
        shuffle=False
    )

    # Création du modèle (Attention U-Net)
    model = attention_unet_model(input_size=(img_width, img_height, 3), num_classes=num_classes)
    model.summary()  # Affiche l'architecture du modèle

    # Gestion des checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt.weights.h5")

    ckpt = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=5)

    # Best model checkpoint
    best_ckpt = tf.train.Checkpoint(model=model)
    best_manager = tf.train.CheckpointManager(best_ckpt, BEST_MODEL_DIR, max_to_keep=1)

    if not args.start_from_scratch and manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"[INFO] Restored from {manager.latest_checkpoint}")
    else:
        print("[INFO] Initializing from scratch...")

    # Callbacks améliorés
    callbacks = [
        # Sauvegarde du meilleur modèle
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(BEST_MODEL_DIR, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        # Sauvegarde régulière des poids
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            monitor='val_loss',
            mode='min'
        ),
        # Arrêt anticipé
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Réduction du taux d'apprentissage sur plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Journal TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]

    # Entraînement du modèle
    history = model.fit(
        train_generator, 
        epochs=args.epochs,
        validation_data=val_generator, 
        callbacks=callbacks,
        verbose=1
    )

    # Sauvegarde du modèle final
    model.save('vertebrae_heart_segmentation_model.h5')
    print("[INFO] Modèle final sauvegardé avec succès!")

if __name__ == "__main__":
    main()