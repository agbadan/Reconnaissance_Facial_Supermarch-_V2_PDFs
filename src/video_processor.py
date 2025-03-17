# src/video_processor.py

import cv2
import numpy as np
import torch
import os
import logging
import tempfile
import zipfile
from datetime import datetime
from tqdm import tqdm
import mediapipe as mp
from face_saver import save_detected_face

from helpers import ensure_directory_exists
from face_tracker import FaceTracker

# --- Gestion de ByteTrack --- 
# Ajout du dossier ByteTrack dans sys.path pour accéder aux modules (nets, utils, etc.)
import sys
bytrack_path = os.path.join(os.getcwd(), "/content/Reconnaissance_Facial_Supermarch-_V2_PDFs/ByteTrack")
if bytrack_path not in sys.path:
    sys.path.append(bytrack_path)

# Importation des modules ByteTrack
from nets import nn
from utils import util  # Attention : ce utils provient de ByteTrack

def crop_face(img: np.ndarray, face_obj: any) -> np.ndarray:
    x1, y1, x2, y2 = face_obj.bbox.astype(int)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    return img[y1:y2, x1:x2]

class VideoProcessor:
    def __init__(self, video_path: str, detection_size: int, face_save_folder: str,
                 log_achat_file: str, log_traite_file: str, config: dict):
        self.video_path = video_path
        self.detection_size = detection_size
        self.face_save_folder = face_save_folder
        ensure_directory_exists(self.face_save_folder)
        self.reader = cv2.VideoCapture(self.video_path)
        if not self.reader.isOpened():
            raise IOError("Erreur lors de l'ouverture de la vidéo.")
        self.fps = int(self.reader.get(cv2.CAP_PROP_FPS))
        self.bytetrack = nn.BYTETracker(self.fps)
        self.util = util
        self.frame_count = 0

        self.log_achat_file = log_achat_file
        self.log_traite_file = log_traite_file
        self.log_achat = open(self.log_achat_file, "a")
        self.log_traite = open(self.log_traite_file, "a")

        self.processing_logs = []
        self.detection_gallery = []
        self.candidates_by_person = {}

        # Chargement du modèle YOLO
        self.model = torch.load('/content/Reconnaissance_Facial_Supermarch-_V2_PDFs/ByteTrack/weights/v8_n.pt', map_location='cuda')['model'].float()
        self.model.eval()
        self.model.half()

        # Initialisation d'InsightFace
        from insightface.app import FaceAnalysis
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Initialisation du suivi des visages
        self.face_tracker = FaceTracker(similarity_threshold=config.get("SIMILARITY_THRESHOLD", 0.45))

        # Initialisation de Mediapipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def generate_face_filename(self, unique_id: str) -> str:
        now_str = datetime.now().strftime("%d_%m_%Y_%H_%M")
        base_filename = f"{unique_id}_{now_str}"
        person_folder = os.path.join(self.face_save_folder, unique_id)
        ensure_directory_exists(person_folder)
        counter = len([f for f in os.listdir(person_folder) if f.startswith(base_filename)]) + 1
        filename = f"{base_filename}_{counter}.jpg"
        return filename

    def log_new_person(self, filename: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{filename} - Nouveau visage détecté enregistré à {now}."
        self.log_achat.write(log_entry + "\n")
        self.log_achat.flush()
        logging.info(log_entry)

    def log_processed_person(self, filename: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{filename} - Visage déjà traité à {now}."
        self.log_traite.write(log_entry + "\n")
        self.log_traite.flush()
        logging.info(log_entry)

    def process(self, face_db: dict) -> tuple:
        total_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        from utils import util  # pour redimensionnement
        for _ in tqdm(range(total_frames), desc="Traitement des frames"):
            ret, frame = self.reader.read()
            if not ret:
                break
            self.frame_count += 1

            # Traiter 1 frame sur 30
            if self.frame_count % 30 != 0:
                continue

            # Filtrage d'orientation avec Mediapipe FaceMesh
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = []
                h_frame, w_frame, _ = frame.shape
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w_frame)
                    y = int(landmark.y * w_frame)
                    z = landmark.z
                    landmarks.append((x, y, z))
                if len(landmarks) > 263:
                    nose_tip = landmarks[1]
                    chin = landmarks[152]
                    left_eye = landmarks[33]
                    right_eye = landmarks[263]
                    roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
                    pitch = np.degrees(np.arctan2(chin[1] - nose_tip[1], chin[2] - nose_tip[2]))
                    yaw = np.degrees(np.arctan2(nose_tip[0] - chin[0], nose_tip[2] - chin[2]))
                    from config import DESIRED_ROLL, DESIRED_PITCH, DESIRED_YAW, ANGLE_TOLERANCE
                    if (abs(roll - DESIRED_ROLL) > ANGLE_TOLERANCE or
                        abs(pitch - DESIRED_PITCH) > ANGLE_TOLERANCE or
                        abs(yaw - DESIRED_YAW) > ANGLE_TOLERANCE):
                        continue
                    score = abs(roll - DESIRED_ROLL) + abs(pitch - DESIRED_PITCH) + abs(yaw - DESIRED_YAW)
                else:
                    continue
            else:
                continue

            # Prétraitement pour ByteTrack
            image = frame.copy()
            original_shape = image.shape[:2]
            r = self.detection_size / max(original_shape)
            if r != 1:
                h, w = original_shape
                image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
            h, w = image.shape[:2]
            image_resized, ratio, pad = self.util.resize(image, self.detection_size)
            shapes_param = original_shape, ((h / original_shape[0], w / original_shape[1]), pad)
            sample = image_resized.transpose((2, 0, 1))[::-1]
            sample = np.ascontiguousarray(sample)
            sample = torch.unsqueeze(torch.from_numpy(sample), 0).cuda().half() / 255.

            # Inférence avec YOLO
            with torch.no_grad():
                outputs = self.model(sample)
            outputs = self.util.non_max_suppression(outputs, 0.001, 0.7)

            boxes = []
            confidences = []
            object_classes = []
            for i, output in enumerate(outputs):
                detections = output.clone()
                self.util.scale(detections[:, :4], sample[i].shape[1:], shapes_param[0], shapes_param[1])
                detections = detections.cpu().numpy()
                for detection in detections:
                    x1, y1, x2, y2 = list(map(int, detection[:4]))
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(detection[4])
                    object_classes.append(int(detection[5]))

            outputs_bt = self.bytetrack.update(np.array(boxes), np.array(confidences), np.array(object_classes))
            if len(outputs_bt) > 0:
                boxes_bt = outputs_bt[:, :4]
                bt_ids = outputs_bt[:, 4]
                obj_classes = outputs_bt[:, 6]
                for i, box in enumerate(boxes_bt):
                    if int(obj_classes[i]) != 0:
                        continue  # ne traiter que la classe "personne"
                    x1, y1, x2, y2 = list(map(int, box))
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi is None or face_roi.size == 0:
                        continue
                    try:
                        detected_faces = self.face_app.get(face_roi)
                    except Exception as e:
                        logging.error(f"Erreur lors de l'analyse du visage: {e}")
                        continue
                    if detected_faces:
                        _, logs_tracker = self.face_tracker.update(detected_faces, int(bt_ids[i]),
                                                                    face_db, "", self.frame_count)
                        for log in logs_tracker:
                            self.processing_logs.append(log)
                        unique_id = detected_faces[0].assigned_label
                        face_crop = crop_face(face_roi, detected_faces[0])
                        if unique_id not in self.candidates_by_person:
                            self.candidates_by_person[unique_id] = []
                        self.candidates_by_person[unique_id].append((score, face_crop))
                        self.processing_logs.append(f"Visage détecté pour {unique_id} (score d'orientation: {score:.2f}).")
                        logging.info(f"Visage détecté pour {unique_id} (score d'orientation: {score:.2f}).")
            # Fin de la boucle sur les frames

        self.reader.release()
        cv2.destroyAllWindows()

        # Sauvegarde des meilleures images pour chaque personne (max 7 par personne)
        for unique_id, candidates in self.candidates_by_person.items():
            candidates.sort(key=lambda x: x[0])
            best_candidates = candidates[:7]
            person_folder = os.path.join(self.face_save_folder, unique_id)
            ensure_directory_exists(person_folder)
            for candidate in best_candidates:
              score, face_crop = candidate
              # Sauvegarder le visage en appelant la fonction commune
              filename, save_path = save_detected_face(face_crop, unique_id, self.face_save_folder)
              self.log_new_person(filename)
              log_msg = f"Visage pour {unique_id} sauvegardé dans {save_path} (score: {score:.2f})."
              self.processing_logs.append(log_msg)
              logging.info(log_msg)
              face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
              self.detection_gallery.append((face_rgb, log_msg))

        self.log_achat.close()
        self.log_traite.close()
        return self.processing_logs, self.detection_gallery
