import os 
import pandas as pd
import shutil

excel_files = "./VERTEBRAE_HEART_SCALES_ANNOTATIONS_MT.xlsx"
annotated_images_folder = "./data"
output_folder = "./filtered_data"

df = pd.read_excel(excel_files)
file_names = df["\nImage file name"].to_list()

# Assurez-vous que la colonne contenant les noms des fichiers est correctement identifiée
os.makedirs(output_folder, exist_ok=True)

# Filtrer les images
for image_file in os.listdir(annotated_images_folder):
    if image_file in file_names:
        # Copier les fichiers correspondants dans le dossier de sortie
        source_path = os.path.join(annotated_images_folder, image_file)
        destination_path = os.path.join(output_folder, image_file)
        shutil.copy(source_path, destination_path)
        print(f"Copied: {image_file}")

print("Filtrage terminé.")