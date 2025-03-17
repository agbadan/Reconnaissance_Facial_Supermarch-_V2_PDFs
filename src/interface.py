# src/interface.py

import gradio as gr
import os
import logging
import time
import cv2
import numpy as np
import torch
import mediapipe as mp
import tempfile
import zipfile

from video_processor import VideoProcessor
from face_database import FaceDatabase
from receipt_generator import ReceiptGenerator, zip_receipts_for_person
from config import (
    DATABASE_PATH,
    FACE_SAVE_FOLDER,
    RECEIPTS_FOLDER,
    DEFAULT_LOG_ACHAT,
    DEFAULT_LOG_TRAITE,
    DETECTION_SIZE,
    SIMILARITY_THRESHOLD,  # Pensez à augmenter ce seuil si nécessaire
    RECEIPT_NUMBER_FILE,
    DESIRED_ROLL,
    DESIRED_PITCH,
    DESIRED_YAW,
    ANGLE_TOLERANCE,
)
from face_saver import save_detected_face
from utils import util  # Module de ByteTrack
from nets import nn       # Module de ByteTrack

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# Création d'une instance unique de la base persistante
# ---------------------------
face_db_obj = FaceDatabase(DATABASE_PATH)
# Cette instance sera utilisée pour les mises à jour en temps réel et lors de l'upload
face_db_shared = face_db_obj.faces_db

# ---------------------------
# Fonctions utilitaires
# ---------------------------
def get_person_ids() -> list:
    if not os.path.exists(FACE_SAVE_FOLDER):
        return []
    return [d for d in os.listdir(FACE_SAVE_FOLDER) if os.path.isdir(os.path.join(FACE_SAVE_FOLDER, d))]

def refresh_person_ids() -> gr.update:
    updated_ids = get_person_ids()
    return gr.update(choices=updated_ids)

def process_video_gradio(video_file) -> tuple:
    video_path = video_file.name if hasattr(video_file, "name") else video_file
    # Utilisation de la même instance de base persistante
    face_db = face_db_shared

    config = {"SIMILARITY_THRESHOLD": SIMILARITY_THRESHOLD}
    processor = VideoProcessor(
        video_path, DETECTION_SIZE, FACE_SAVE_FOLDER,
        DEFAULT_LOG_ACHAT, DEFAULT_LOG_TRAITE, config
    )
    logs_list, gallery = processor.process(face_db)

    # Mise à jour et sauvegarde de la base persistante
    face_db_obj.faces_db = face_db
    face_db_obj.save()

    # Génération des reçus pour chaque visage détecté
    receipt_gen = ReceiptGenerator(
        supermarket_name="Supermarché le champion",
        supermarket_address="Agoe Demakpoe",
        supermarket_tel="+228 99520033",
        face_save_folder=FACE_SAVE_FOLDER,
        receipts_folder=RECEIPTS_FOLDER,
        default_log=DEFAULT_LOG_ACHAT,
        receipt_number_file=RECEIPT_NUMBER_FILE
    )
    receipt_gen.process_all_new_receipts()

    updated_ids = get_person_ids()
    return "\n".join(logs_list), gallery, gr.update(choices=updated_ids)

def download_receipts_gradio(unique_id: str) -> str:
    zip_path = zip_receipts_for_person(unique_id, RECEIPTS_FOLDER)
    if not zip_path:
        # Création d'un fichier zip temporaire contenant un message
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        temp_zip.close()
        with zipfile.ZipFile(temp_zip.name, 'w') as zf:
            message = f"Aucun reçu trouvé pour {unique_id}."
            zf.writestr("message.txt", message)
        return temp_zip.name
    return zip_path

# ---------------------------
# Détection en temps réel via webcam
# ---------------------------
# Initialisation de Mediapipe FaceMesh pour le temps réel
mp_face_mesh_rt = mp.solutions.face_mesh
face_mesh_rt = mp_face_mesh_rt.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Chargement du modèle YOLO pour le temps réel et initialisation de ByteTrack
model_rt = torch.load('./weights/v8_n.pt', map_location='cuda')['model'].float()
model_rt.eval()
model_rt.half()
bytetrack_rt = nn.BYTETracker(30)  # On fixe 30 fps pour ByteTrack

# Variables globales pour la détection en temps réel
last_time_rt = 0
last_output_rt = None
gallery_rt = []  # Galerie locale pour stocker les visages détectés

# Pour la reconnaissance en temps réel, on charge InsightFace et instancie le FaceTracker
from insightface.app import FaceAnalysis
face_app_rt = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
face_app_rt.prepare(ctx_id=0, det_size=(640, 640))

from face_tracker import FaceTracker
# Utilisation de la même base persistante partagée
face_tracker_rt = FaceTracker(similarity_threshold=SIMILARITY_THRESHOLD)
# On assigne la référence partagée à l'instance du tracker
face_db_rt = face_db_shared

# Variable pour limiter la fréquence d'appel à la génération des reçus
last_receipt_generation_rt = 0
RECEIPT_GEN_INTERVAL = 5  # en secondes

# Historique persistant des logs en temps réel
rt_logs_history = []

def process_realtime_detection(frame):
    """
    Traite une frame de la webcam en temps réel :
      - Traite au maximum une frame par seconde.
      - Vérifie l'orientation via Mediapipe.
      - Exécute le pipeline YOLO/ByteTrack pour détecter les visages.
      - Pour chaque visage détecté, extrait la région du visage, réalise la reconnaissance via InsightFace,
        et met à jour le tracker en utilisant la base partagée.
      - Le visage est sauvegardé via save_detected_face et ajouté à la galerie.
      - Périodiquement, déclenche la génération des reçus.
      - Les logs détaillés sont ajoutés à un historique persistant pour l'affichage.
    Retourne un tuple (frame annotée, galerie des visages détectés, logs persistant).
    """
    global last_time_rt, last_output_rt, gallery_rt, face_db_rt, last_receipt_generation_rt, rt_logs_history
    rt_logs = []  # logs de cette frame
    frame = frame.copy()
    current_time = time.time()
    if current_time - last_time_rt < 1:
        return last_output_rt if last_output_rt is not None else frame, gallery_rt, "\n".join(rt_logs_history)
    last_time_rt = current_time

    # 1. Détection via Mediapipe FaceMesh
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_rt.process(frame_rgb)
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
            rt_logs.append(f"Angles: roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
            if (abs(roll - DESIRED_ROLL) > ANGLE_TOLERANCE or
                abs(pitch - DESIRED_PITCH) > ANGLE_TOLERANCE or
                abs(yaw - DESIRED_YAW) > ANGLE_TOLERANCE):
                rt_logs.append("Angles non optimaux, mais on continue le traitement.")
        else:
            rt_logs.append("Nombre insuffisant de landmarks, on continue.")
    else:
        rt_logs.append("Aucun landmark détecté, on continue.")

    # 2. Prétraitement pour YOLO et détection avec ByteTrack
    image = frame.copy()
    original_shape = image.shape[:2]
    r = DETECTION_SIZE / max(original_shape)
    if r != 1:
        h, w = original_shape
        image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    h, w = image.shape[:2]
    image_resized, ratio, pad = util.resize(image, DETECTION_SIZE)
    shapes_param = (original_shape, ((h / original_shape[0], w / original_shape[1]), pad))
    sample = image_resized.transpose((2, 0, 1))[::-1]
    sample = np.ascontiguousarray(sample)
    sample = torch.unsqueeze(torch.from_numpy(sample), 0).cuda().half() / 255.
    
    with torch.no_grad():
        outputs = model_rt(sample)
    outputs = util.non_max_suppression(outputs, 0.001, 0.7)

    boxes, confidences, object_classes = [], [], []
    for i, output in enumerate(outputs):
        detections = output.clone()
        util.scale(detections[:, :4], sample[i].shape[1:], shapes_param[0], shapes_param[1])
        detections = detections.cpu().numpy()
        for detection in detections:
            x1, y1, x2, y2 = list(map(int, detection[:4]))
            boxes.append([x1, y1, x2, y2])
            confidences.append(detection[4])
            object_classes.append(int(detection[5]))
    
    outputs_bt = bytetrack_rt.update(np.array(boxes), np.array(confidences), np.array(object_classes))
    if len(outputs_bt) > 0:
        boxes_bt = outputs_bt[:, :4]
        obj_classes = outputs_bt[:, 6]
        for i, box in enumerate(boxes_bt):
            if int(obj_classes[i]) != 0:
                continue  # Ne traiter que la classe "personne"
            x1, y1, x2, y2 = list(map(int, box))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Extraction du face crop
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size != 0:
                try:
                    detected_faces = face_app_rt.get(face_roi)
                except Exception as e:
                    rt_logs.append(f"Erreur lors de l'analyse du visage : {e}")
                    detected_faces = []
                if detected_faces:
                    # Mise à jour du tracker avec la base partagée
                    _, logs_tracker = face_tracker_rt.update(detected_faces, int(outputs_bt[i, 4]), face_db_rt, "", 0)
                    for log in logs_tracker:
                        rt_logs.append(log)
                    unique_id = detected_faces[0].assigned_label
                    rt_logs.append(f"Visage reconnu : {unique_id}")
                else:
                    unique_id = "unknown"
                    rt_logs.append("Aucun visage reconnu, identifiant 'unknown'")
                # Sauvegarde de la région du visage et mise à jour de la base persistante
                filename, save_path = save_detected_face(face_roi, unique_id, FACE_SAVE_FOLDER)
                rt_logs.append(f"Face saved: {filename} at {save_path}")
                gallery_rt.append(face_roi)
                if len(gallery_rt) > 10:
                    gallery_rt.pop(0)
                # Synchronisation immédiate de la base partagée
                face_db_obj.faces_db = face_db_rt
                face_db_obj.save()
    
    # Appel périodique à la génération des reçus (toutes les 5 secondes)
    if time.time() - last_receipt_generation_rt > RECEIPT_GEN_INTERVAL:
        receipt_gen = ReceiptGenerator(
            supermarket_name="Supermarché le champion",
            supermarket_address="Agoe Demakpoe",
            supermarket_tel="+228 99520033",
            face_save_folder=FACE_SAVE_FOLDER,
            receipts_folder=RECEIPTS_FOLDER,
            default_log=DEFAULT_LOG_ACHAT,
            receipt_number_file=RECEIPT_NUMBER_FILE
        )
        receipt_gen.process_all_new_receipts()
        last_receipt_generation_rt = time.time()
        rt_logs.append("Génération des reçus déclenchée.")

    # Ajout des logs de cette frame à l'historique global
    rt_logs_history.extend(rt_logs)
    # Limiter l'historique aux 100 derniers messages
    if len(rt_logs_history) > 100:
        rt_logs_history = rt_logs_history[-100:]
    
    last_output_rt = frame
    return frame, gallery_rt, "\n".join(rt_logs_history)

# ---------------------------
# Interface Gradio – Définition des onglets
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Reconnaissance Faciale – Traitement Vidéo, Détection en Temps Réel, Génération Instantanée de Reçus et Téléchargement")
    with gr.Tabs():
        with gr.TabItem("Traitement Vidéo"):
            video_input = gr.Video(label="Uploader votre vidéo")
            logs_output = gr.Textbox(label="Logs de traitement", lines=10)
            gallery_output = gr.Gallery(label="Galerie de visages détectés")
            updated_dropdown = gr.Dropdown(choices=get_person_ids(), label="Identifiants mis à jour")
            process_btn = gr.Button("Lancer le traitement")
            process_btn.click(fn=process_video_gradio,
                              inputs=video_input,
                              outputs=[logs_output, gallery_output, updated_dropdown])
        with gr.TabItem("Détection en temps réel"):
            realtime_img = gr.Image(source="webcam", streaming=True, label="Flux en temps réel")
            realtime_gallery = gr.Gallery(label="Galerie des visages détectés")
            rt_logs_output = gr.Textbox(label="Logs détection temps réel", lines=8)
            realtime_img.stream(fn=process_realtime_detection,
                                inputs=realtime_img,
                                outputs=[realtime_img, realtime_gallery, rt_logs_output])
        with gr.TabItem("Télécharger Recus"):
            person_dropdown = gr.Dropdown(choices=get_person_ids(), label="Sélectionnez un identifiant", interactive=True)
            refresh_btn = gr.Button("Rafraîchir la liste")
            download_btn = gr.Button("Télécharger les reçus")
            download_output = gr.File(label="Fichier ZIP des reçus")
            refresh_btn.click(fn=refresh_person_ids, inputs=[], outputs=person_dropdown)
            download_btn.click(fn=download_receipts_gradio, inputs=person_dropdown, outputs=download_output)

demo.launch(debug=True, share=True)
