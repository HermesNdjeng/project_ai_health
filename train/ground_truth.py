import os
import cv2
import numpy as np
import argparse

def generate_ground_truth(image_dir, annotation_dir, output_dir, img_width=512, img_height=512):
    """
    Génère les masques de vérité terrain à partir des annotations.
    
    Args:
        image_dir (str): Répertoire des images
        annotation_dir (str): Répertoire des annotations (au format YOLO)
        output_dir (str): Répertoire de sortie pour les masques
        img_width (int): Largeur des images de sortie
        img_height (int): Hauteur des images de sortie
    """
    print(f"Génération des masques de vérité terrain pour {annotation_dir}...")
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    class_colors = {
        0: (255, 0, 0),  # Rouge pour la classe 0
        1: (0, 255, 0),  # Vert pour la classe 1
        2: (0, 0, 255),  # Bleu pour la classe 2
    }
    
    processed_count = 0
    total_annotations = len([f for f in os.listdir(annotation_dir) if f.endswith(".txt")])
    
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
            
            cv2.imwrite(output_path, ground_truth)
            processed_count += 1

            if processed_count % 100 == 0:
                print(f"Progression: {processed_count}/{total_annotations} annotations traitées")
    
    print(f"Génération terminée. {processed_count} masques de vérité terrain générés dans {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Génère des masques de vérité terrain à partir d'annotations YOLO")
    parser.add_argument("--train-only", action="store_true", help="Traiter uniquement les données d'entraînement")
    parser.add_argument("--valid-only", action="store_true", help="Traiter uniquement les données de validation")
    args = parser.parse_args()
    
    # Chemins pour l'ensemble d'entraînement
    train_image_dir = "vertebrae-yolo-dataset/train/images"
    train_annotation_dir = "vertebrae-yolo-dataset/train/labels"
    train_output_dir = "vertebrae-yolo-dataset/train/ground_truth_annotated"
    
    # Chemins pour l'ensemble de validation
    valid_image_dir = "vertebrae-yolo-dataset/valid/images"
    valid_annotation_dir = "vertebrae-yolo-dataset/valid/labels"
    valid_output_dir = "vertebrae-yolo-dataset/valid/ground_truth_annotated"
    
    # Génération des masques selon les paramètres
    if not args.valid_only:
        generate_ground_truth(train_image_dir, train_annotation_dir, train_output_dir)
    
    if not args.train_only:
        generate_ground_truth(valid_image_dir, valid_annotation_dir, valid_output_dir)
    
    print("Traitement terminé!")

if __name__ == "__main__":
    main()