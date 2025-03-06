import os
import shutil

def delete_unannotated_images(image_dir, annotation_dir):
    """
    Supprime les images dans image_dir qui n'ont pas d'annotations correspondantes dans annotation_dir.
    
    Args:
        image_dir (str): Répertoire contenant les fichiers image
        annotation_dir (str): Répertoire contenant les fichiers d'annotation
    """
    print(f"Vérification des images sans annotation dans {image_dir}...")
    
    # Récupérer tous les fichiers d'image
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Nombre total d'images: {len(image_files)}")
    
    # Récupérer tous les fichiers d'annotation
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]
    print(f"Nombre total d'annotations: {len(annotation_files)}")
    
    # Extraire les noms de base sans extension
    annotation_basenames = set([os.path.splitext(f)[0] for f in annotation_files])
    
    deleted_count = 0
    for image_file in image_files:
        # Obtenir le nom de base (sans extension) du fichier image
        image_basename = os.path.splitext(image_file)[0]
        
        # Vérifier s'il existe un fichier d'annotation correspondant
        if image_basename not in annotation_basenames:
            image_path = os.path.join(image_dir, image_file)
            print(f"Suppression de {image_path} (aucune annotation trouvée)")
            os.remove(image_path)
            deleted_count += 1
    
    print(f"Suppression de {deleted_count} images sans annotations terminée.")

def main():
    # Chemins pour l'ensemble d'entraînement
    train_image_dir = "vertebrae-yolo-dataset/train/images"
    train_annotation_dir = "vertebrae-yolo-dataset/train/labels"
    
    # Vérifier et supprimer les images sans annotations dans l'ensemble d'entraînement
    delete_unannotated_images(train_image_dir, train_annotation_dir)
    
    # Chemins pour l'ensemble de validation
    valid_image_dir = "vertebrae-yolo-dataset/valid/images"
    valid_annotation_dir = "vertebrae-yolo-dataset/valid/labels"
    
    # Vérifier et supprimer les images sans annotations dans l'ensemble de validation
    delete_unannotated_images(valid_image_dir, valid_annotation_dir)
    
    print("Nettoyage terminé!")

if __name__ == "__main__":
    main()