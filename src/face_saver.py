# face_saver.py
import os
import cv2
from datetime import datetime

def save_detected_face(face_roi, unique_id, face_save_folder):
    """
    Sauvegarde l'image du visage (face_roi) dans le dossier correspondant à l'identifiant (unique_id).
    Retourne le nom du fichier généré et le chemin complet.
    """
    now_str = datetime.now().strftime("%d_%m_%Y_%H_%M")
    base_filename = f"{unique_id}_{now_str}"
    person_folder = os.path.join(face_save_folder, unique_id)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    # Compte le nombre d'images déjà présentes
    counter = len([f for f in os.listdir(person_folder) if f.startswith(base_filename)]) + 1
    filename = f"{base_filename}_{counter}.jpg"
    save_path = os.path.join(person_folder, filename)
    cv2.imwrite(save_path, face_roi)
    return filename, save_path
