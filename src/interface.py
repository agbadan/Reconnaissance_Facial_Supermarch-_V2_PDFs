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
    TORCH_WEIGHT_PATH,
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
    face_db = face_db_shared

    config = {"SIMILARITY_THRESHOLD": SIMILARITY_THRESHOLD}
    processor = VideoProcessor(
        video_path, DETECTION_SIZE, FACE_SAVE_FOLDER,
        DEFAULT_LOG_ACHAT, DEFAULT_LOG_TRAITE, config
    )
    logs_list, gallery = processor.process(face_db)

    face_db_obj.faces_db = face_db
    face_db_obj.save()

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
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        temp_zip.close()
        with zipfile.ZipFile(temp_zip.name, 'w') as zf:
            message = f"Aucun reçu trouvé pour {unique_id}."
            zf.writestr("message.txt", message)
        return temp_zip.name
    return zip_path

# ---------------------------
# Détection en temps réel via webcam (local)
# ---------------------------
mp_face_mesh_rt = mp.solutions.face_mesh
face_mesh_rt = mp_face_mesh_rt.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

model_rt = torch.load(TORCH_WEIGHT_PATH, map_location='cuda')['model'].float()
model_rt.eval()
model_rt.half()
bytetrack_rt = nn.BYTETracker(30)

last_time_rt = 0
last_output_rt = None
gallery_rt = []

from insightface.app import FaceAnalysis
face_app_rt = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
face_app_rt.prepare(ctx_id=0, det_size=(640, 640))

from face_tracker import FaceTracker
face_tracker_rt = FaceTracker(similarity_threshold=SIMILARITY_THRESHOLD)
face_db_rt = face_db_shared

last_receipt_generation_rt = 0
RECEIPT_GEN_INTERVAL = 5

rt_logs_history = []

def process_realtime_detection(frame):
    """
    Traite une frame locale en temps réel et renvoie l'image annotée (RGB), la galerie des visages détectés et les logs.
    """
    global last_time_rt, last_output_rt, gallery_rt, face_db_rt, last_receipt_generation_rt, rt_logs_history
    rt_logs = []
    # Conversion de l'image d'entrée de RGB (format Gradio) en BGR (format OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = frame.copy()
    current_time = time.time()
    if current_time - last_time_rt < 1:
        last_output_rgb = cv2.cvtColor(last_output_rt, cv2.COLOR_BGR2RGB) if last_output_rt is not None else None
        return last_output_rgb, gallery_rt, "\n".join(rt_logs_history)
    last_time_rt = current_time

    # Détection avec Mediapipe (Mediapipe attend du RGB, on convertit donc temporairement)
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
            pitch = np.degrees(np.arctan2(chin[1] - nose_tip[1], chin[2] - nose_tip[1]))
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

    # Détection avec YOLO / ByteTrack
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
                continue
            x1, y1, x2, y2 = list(map(int, box))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size != 0:
                try:
                    detected_faces = face_app_rt.get(face_roi)
                except Exception as e:
                    rt_logs.append(f"Erreur lors de l'analyse du visage : {e}")
                    detected_faces = []
                if detected_faces:
                    _, logs_tracker = face_tracker_rt.update(detected_faces, int(outputs_bt[i, 4]), face_db_rt, "", 0)
                    for log in logs_tracker:
                        rt_logs.append(log)
                    unique_id = detected_faces[0].assigned_label
                    rt_logs.append(f"Visage reconnu : {unique_id}")
                else:
                    unique_id = "unknown"
                    rt_logs.append("Aucun visage reconnu, identifiant 'unknown'")
                filename, save_path = save_detected_face(face_roi, unique_id, FACE_SAVE_FOLDER)
                rt_logs.append(f"Face saved: {filename} at {save_path}")
                # Convertir le visage en RGB avant de l'ajouter à la galerie
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                gallery_rt.append(face_roi_rgb)
                if len(gallery_rt) > 10:
                    gallery_rt.pop(0)
                face_db_obj.faces_db = face_db_rt
                face_db_obj.save()
                cv2.putText(frame, unique_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
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

    rt_logs_history.extend(rt_logs)
    if len(rt_logs_history) > 100:
        rt_logs_history = rt_logs_history[-100:]
    
    last_output_rt = frame
    # Conversion finale en RGB pour l'affichage dans Gradio
    output_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return output_rgb, gallery_rt, "\n".join(rt_logs_history)

# ---------------------------
# Détection via flux distant (URL) – fonction génératrice
# ---------------------------
def stream_remote_detection(url):
    cap = cv2.VideoCapture(url)
    frame_count = 0
    last_output = None
    if not cap.isOpened():
        yield None, [], f"Erreur : Impossible d'ouvrir le flux à l'URL {url}"
    while cap.isOpened():
        # Vider le tampon pour ne traiter que la frame la plus récente
        for _ in range(5):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            yield None, [], "Fin du flux ou erreur de lecture."
            break
        frame_count += 1
        # Ne traiter qu'une frame sur 30
        if frame_count % 30 != 0:
            if last_output is not None:
                yield last_output
            continue
        processed_frame, gallery, logs = process_realtime_detection(frame)
        last_output = (processed_frame, gallery, logs)
        yield processed_frame, gallery, logs
    cap.release()

# ---------------------------
# Interface Gradio – Définition des onglets
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Reconnaissance Faciale – Traitement Vidéo, Détection en Temps Réel, Détection via URL, et Téléchargement de Recus")
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
            realtime_img = gr.Image(sources="webcam", streaming=True, label="Flux en temps réel")
            realtime_gallery = gr.Gallery(label="Galerie des visages détectés")
            rt_logs_output = gr.Textbox(label="Logs détection temps réel", lines=8)
            realtime_img.stream(fn=process_realtime_detection,
                                inputs=realtime_img,
                                outputs=[realtime_img, realtime_gallery, rt_logs_output])
        with gr.TabItem("Détection via URL"):
            url_input = gr.Textbox(label="Entrez l'URL du flux distant", placeholder="https://2861-102-64-160-170.ngrok-free.app/video")
            remote_img = gr.Image(label="Flux distant traité", streaming=True)
            remote_gallery = gr.Gallery(label="Galerie des visages détectés")
            remote_logs = gr.Textbox(label="Logs de détection distante", lines=8)
            start_btn = gr.Button("Lancer la détection")
            start_btn.click(fn=stream_remote_detection,
                            inputs=url_input,
                            outputs=[remote_img, remote_gallery, remote_logs])
        with gr.TabItem("Télécharger Recus"):
            person_dropdown = gr.Dropdown(choices=get_person_ids(), label="Sélectionnez un identifiant", interactive=True)
            refresh_btn = gr.Button("Rafraîchir la liste")
            download_btn = gr.Button("Télécharger les reçus")
            download_output = gr.File(label="Fichier ZIP des reçus")
            refresh_btn.click(fn=refresh_person_ids, inputs=[], outputs=person_dropdown)
            download_btn.click(fn=download_receipts_gradio, inputs=person_dropdown, outputs=download_output)

demo.launch(debug=True, share=True)
