import cv2
import numpy as np

# Chemins des dossiers
image_dir = "vertebrae-yolo-dataset/train/images"  # Dossier contenant les images originales
annotation_dir = "vertebrae-yolo-dataset/train/labels"  # Dossier contenant les annotations
output_dir = "vertebrae-yolo-dataset/train/ground_truth"  # Dossier pour sauvegarder les images binaires

import os
import cv2
import numpy as np

# Dimensions des images (par exemple, 1024x1024)
img_width = 1024
img_height = 1024

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Parcourir tous les fichiers d'annotations dans le dossier
for annotation_file in os.listdir(annotation_dir):
    if annotation_file.endswith(".txt"):
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        # Nom de l'image correspondante
        image_name = os.path.splitext(annotation_file)[0] + ".png"
        output_path = os.path.join(output_dir, image_name)
        
        # Image vide (noire)
        ground_truth = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Lecture des annotations
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  # Classe (non utilisée ici)
            coordinates = list(map(float, parts[1:]))
            
            # Conversion des coordonnées normalisées en pixels
            points = [
                (int(x * img_width), int(y * img_height))
                for x, y in zip(coordinates[0::2], coordinates[1::2])
            ]
            
            # Dessiner le polygone sur l'image (rempli en blanc)
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(ground_truth, [points], color=255)
        
        # Sauvegarder l'image de ground truth
        cv2.imwrite(output_path, ground_truth)
        print(f"Image de ground truth générée : {output_path}")
