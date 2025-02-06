import os
import cv2
import numpy as np

# Chemins des dossiers
image_dir = "vertebrae-yolo-dataset/train/images"  
annotation_dir = "vertebrae-yolo-dataset/train/labels"  
output_dir = "vertebrae-yolo-dataset/train/ground_truth_annotated"  

# Dimensions des images (par exemple, 1024x1024)
img_width = 512
img_height = 512

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

class_colors = {
    0: (255, 0, 0),  # Rouge pour la classe 0
    1: (0, 255, 0),  # Vert pour la classe 1
    2: (0, 0, 255),  # Bleu pour la classe 2
}

# Parcourir tous les fichiers d'annotations dans le dossier
for annotation_file in os.listdir(annotation_dir):
    if annotation_file.endswith(".txt"):
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        # Nom de l'image correspondante
        image_name = os.path.splitext(annotation_file)[0] + ".png"
        output_path = os.path.join(output_dir, image_name)
        
        # Image vide (noire)
        ground_truth = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Lecture des annotations
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  # Classe
            coordinates = list(map(float, parts[1:]))
            
            # Conversion des coordonnées normalisées en pixels
            points = [
                (int(x * img_width), int(y * img_height))
                for x, y in zip(coordinates[0::2], coordinates[1::2])
            ]
            
            # Dessiner le polygone sur l'image avec la couleur de la classe
            points = np.array(points, dtype=np.int32)
            color = class_colors.get(class_id, (255, 255, 255))  # Blanc par défaut si la classe n'est pas définie
            cv2.fillPoly(ground_truth, [points], color=color)
        
        # Sauvegarder l'image de ground truth
        cv2.imwrite(output_path, ground_truth)
        print(f"Image de ground truth générée : {output_path}")