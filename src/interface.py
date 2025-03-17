# src/interface.py

import gradio as gr
from video_processor import VideoProcessor
from face_database import FaceDatabase
from receipt_generator import ReceiptGenerator, zip_receipts_for_person
import os
from config import DATABASE_PATH, FACE_SAVE_FOLDER, RECEIPTS_FOLDER, DEFAULT_LOG_ACHAT, DEFAULT_LOG_TRAITE, DETECTION_SIZE, SIMILARITY_THRESHOLD, RECEIPT_NUMBER_FILE
import logging

def get_person_ids() -> list:
    if not os.path.exists(FACE_SAVE_FOLDER):
        return []
    return [d for d in os.listdir(FACE_SAVE_FOLDER) if os.path.isdir(os.path.join(FACE_SAVE_FOLDER, d))]

def refresh_person_ids() -> gr.update:
    updated_ids = get_person_ids()
    return gr.update(choices=updated_ids)

def process_video_gradio(video_file) -> tuple:
    video_path = video_file.name if hasattr(video_file, "name") else video_file
    face_db_obj = FaceDatabase(DATABASE_PATH)
    face_db = face_db_obj.faces_db

    config = {
        "SIMILARITY_THRESHOLD": SIMILARITY_THRESHOLD
    }
    processor = VideoProcessor(video_path, DETECTION_SIZE, FACE_SAVE_FOLDER,
                               DEFAULT_LOG_ACHAT, DEFAULT_LOG_TRAITE, config)
    logs_list, gallery = processor.process(face_db)

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
        return f"Aucun reçu trouvé pour {unique_id}."
    return zip_path

with gr.Blocks() as demo:
    gr.Markdown("## Reconnaissance Faciale – Traitement Vidéo, Génération Instantanée de Reçus et Téléchargement")
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
        with gr.TabItem("Télécharger Recus"):
            person_dropdown = gr.Dropdown(choices=get_person_ids(), label="Sélectionnez un identifiant", interactive=True)
            refresh_btn = gr.Button("Rafraîchir la liste")
            download_btn = gr.Button("Télécharger les reçus")
            download_output = gr.File(label="Fichier ZIP des reçus")
            refresh_btn.click(fn=refresh_person_ids, inputs=[], outputs=person_dropdown)
            download_btn.click(fn=download_receipts_gradio, inputs=person_dropdown, outputs=download_output)

demo.launch(debug=True)
