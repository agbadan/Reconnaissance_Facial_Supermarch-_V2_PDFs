# src/receipt_generator.py

import os
import re
import cv2
import numpy as np
import logging
import zipfile
import tempfile
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import mediapipe as mp
# src/helpers.py
from helpers import ensure_directory_exists

class ReceiptGenerator:
    def __init__(self,
                 supermarket_name: str = "Supermarché le champion",
                 supermarket_address: str = "Agoe Demakpoe",
                 supermarket_tel: str = "+228 99520033",
                 face_save_folder: str = None,
                 receipts_folder: str = None,
                 default_log: str = None,
                 receipt_number_file: str = None):
        self.supermarket_name = supermarket_name
        self.supermarket_address = supermarket_address
        self.supermarket_tel = supermarket_tel
        self.face_save_folder = face_save_folder
        self.receipts_folder = receipts_folder
        self.default_log = default_log
        self.receipt_number_file = receipt_number_file

    def ensure_directory_exists(self, directory: str):
        ensure_directory_exists(directory)

    def get_next_receipt_number(self) -> int:
        last_number = 0
        if os.path.exists(self.receipt_number_file):
            try:
                with open(self.receipt_number_file, "r") as f:
                    last_number = int(f.read().strip())
            except Exception as e:
                logging.error(f"Erreur lors de la lecture du numéro de reçu: {e}")
                last_number = 0
        next_number = last_number + 1
        try:
            with open(self.receipt_number_file, "w") as f:
                f.write(str(next_number))
        except Exception as e:
            logging.error(f"Erreur lors de l'écriture du nouveau numéro de reçu: {e}")
        return next_number

    def receipt_generated_in_current_hour_from_log(self, unique_id: str, receipt_date: datetime) -> bool:
        if not os.path.exists(self.default_log):
            return False
        target_hour = receipt_date.strftime("%Y-%m-%d %H")
        try:
            with open(self.default_log, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        log_uid, log_datetime = line.split("|")
                    except Exception:
                        continue
                    if log_uid == unique_id and log_datetime[:13] == target_hour:
                        return True
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du log: {e}")
        return False

    def select_best_face_image_by_mediapipe(self, person_folder: str) -> tuple:
        DESIRED_ROLL = -0.84
        DESIRED_PITCH = 89.97
        DESIRED_YAW = -91.44

        mp_face_mesh = mp.solutions.face_mesh
        best_score = None
        best_image_path = None
        best_date = None

        with mp_face_mesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=1,
                                   min_detection_confidence=0.5) as face_mesh:
            for file in os.listdir(person_folder):
                if not file.lower().endswith(".jpg"):
                    continue
                image_path = os.path.join(person_folder, file)
                image = cv2.imread(image_path)
                if image is None:
                    logging.error(f"Erreur de lecture de l'image : {image_path}")
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                if not results.multi_face_landmarks:
                    continue
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = image.shape
                try:
                    nose_tip = face_landmarks.landmark[1]
                    chin = face_landmarks.landmark[152]
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]
                except IndexError:
                    continue
                nose_tip = (nose_tip.x * w, nose_tip.y * h)
                chin = (chin.x * w, chin.y * h)
                left_eye = (left_eye.x * w, left_eye.y * h)
                right_eye = (right_eye.x * w, right_eye.y * h)
                roll = np.degrees(np.arctan2(right_eye[1]-left_eye[1], right_eye[0]-left_eye[0]))
                pitch = np.degrees(np.arctan2(chin[1]-nose_tip[1], chin[0]-nose_tip[0]))
                yaw = np.degrees(np.arctan2(nose_tip[0]-chin[0], chin[1]-nose_tip[1]))
                score = abs(roll - DESIRED_ROLL) + abs(pitch - DESIRED_PITCH) + abs(yaw - DESIRED_YAW)
                if best_score is None or score < best_score:
                    best_score = score
                    best_image_path = image_path
                    match = re.search(r'personne_\w+_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})', file)
                    if match:
                        try:
                            day, month, year, hour, minute = match.groups()
                            date_str = f"{day}_{month}_{year}_{hour}_{minute}"
                            best_date = datetime.strptime(date_str, "%d_%m_%Y_%H_%M")
                        except Exception as e:
                            logging.error(f"Erreur lors de l'extraction de la date pour {file} : {e}")
                            best_date = datetime.now()
                    else:
                        logging.warning(f"Format de nom de fichier inattendu : {file}")
                        best_date = datetime.now()
        return best_image_path, best_score, best_date

    def generate_random_items(self) -> list:
        sample_items = [
            "Pomme", "Banane", "Lait", "Pain", "Yaourt", "Fromage",
            "Jus de fruit", "Café", "Thé", "Chocolat", "Bonbons"
        ]
        items = []
        nb_items = np.random.randint(3, 7)
        for _ in range(nb_items):
            name = np.random.choice(sample_items)
            quantity = np.random.randint(1, 11)
            unit_price = round(np.random.uniform(0.5, 5.0) * 100, 2)
            total = round(quantity * unit_price, 2)
            items.append({
                "name": name,
                "quantity": quantity,
                "unit_price": unit_price,
                "total": total
            })
        return items

    def generate_receipt(self, person_folder: str, unique_id: str, best_image_path: str, receipt_date: datetime) -> str:
        receipt_number = self.get_next_receipt_number()
        items = self.generate_random_items()
        total = sum(item["total"] for item in items)

        receipt_filename = f"RECU_{unique_id}_{receipt_date.strftime('%d_%m_%Y_%H_%M')}_N{receipt_number}.pdf"
        self.ensure_directory_exists(self.receipts_folder)
        output_file = os.path.join(self.receipts_folder, receipt_filename)

        c = canvas.Canvas(output_file, pagesize=letter)
        width, height = letter

        # En-tête du reçu
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width/2, height - 50, self.supermarket_name)
        c.setFont("Helvetica", 12)
        c.drawCentredString(width/2, height - 70, f"Adresse : {self.supermarket_address}")
        c.drawCentredString(width/2, height - 85, f"Téléphone : {self.supermarket_tel}")

        # Date et numéro du reçu
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 120, f"Date : {receipt_date.strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, 30, f"Reçu N°: {receipt_number}")

        # Insertion de l'image du visage
        image_width, image_height = 80, 100
        image_x = width - 50 - image_width
        image_y = height - 150
        try:
            c.drawImage(best_image_path, image_x, image_y, width=image_width, height=image_height, preserveAspectRatio=True)
        except Exception as e:
            logging.error(f"Erreur lors de l'insertion de l'image {best_image_path} : {e}")

        # Détail des articles achetés
        c.setFont("Helvetica-Bold", 12)
        y_table = height - 160
        c.drawString(50, y_table, "Article")
        c.drawString(250, y_table, "Quantité")
        c.drawString(350, y_table, "Prix Unitaire")
        c.drawString(450, y_table, "Total")

        c.setFont("Helvetica", 12)
        y = y_table - 20
        for item in items:
            c.drawString(50, y, item['name'])
            c.drawString(250, y, str(item['quantity']))
            c.drawString(350, y, f"{item['unit_price']:.2f} CFA")
            c.drawString(450, y, f"{item['total']:.2f} CFA")
            y -= 20

        # Total et message de remerciement
        y_total = y - 20
        c.setFont("Helvetica-Bold", 12)
        c.drawRightString(450, y_total, "Total :")
        c.drawString(460, y_total, f"{total:.2f} CFA")
        c.setFont("Helvetica", 10)
        c.drawCentredString(width/2, 20, "Merci pour votre achat !")

        c.showPage()
        c.save()
        log_msg = f"Reçu généré pour {unique_id} et sauvegardé dans {output_file}."
        logging.info(log_msg)
        try:
            with open(self.default_log, "a") as f_log:
                f_log.write(f"{unique_id}|{receipt_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            logging.error(f"Erreur lors de l'écriture du log du reçu : {e}")
        return log_msg

    def process_all_new_receipts(self):
        """
        Pour chaque personne détectée, vérifie si un reçu doit être généré.
        Un reçu est généré si aucun n'a été enregistré pour la même heure.
        """
        ensure_directory_exists(self.face_save_folder)
        for unique_id in os.listdir(self.face_save_folder):
            person_folder = os.path.join(self.face_save_folder, unique_id)
            if not os.path.isdir(person_folder):
                continue
            best_image_path, score, receipt_date = self.select_best_face_image_by_mediapipe(person_folder)
            if best_image_path is None:
                logging.warning(f"Aucune image valide trouvée pour {unique_id}. Aucun reçu ne sera généré.")
                continue
            if self.receipt_generated_in_current_hour_from_log(unique_id, receipt_date):
                logging.info(f"Pour {unique_id} : Un reçu a déjà été généré cette heure. Aucun nouveau reçu ne sera créé.")
                continue
            else:
                log_msg = self.generate_receipt(person_folder, unique_id, best_image_path, receipt_date)
                logging.info(f"Pour {unique_id} : {log_msg}")

def zip_receipts_for_person(unique_id: str, receipts_folder: str) -> str:
    person_receipts = []
    if not os.path.exists(receipts_folder):
        return ""
    for file in os.listdir(receipts_folder):
        if unique_id in file:
            person_receipts.append(os.path.join(receipts_folder, file))
    if not person_receipts:
        return ""
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        for filepath in person_receipts:
            zipf.write(filepath, os.path.basename(filepath))
    return temp_zip.name
