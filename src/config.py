# src/config.py

import os

#Chemin vers ByteTrack
BYTETRACK_PATH = os.path.join(os.getcwd(), "/content/drive/MyDrive/Reconnaissance_Facial_Supermarch-_V2_PDFs/ByteTrack/")


#Poids de Bytetrack
TORCH_WEIGHT_PATH = os.path.join(os.getcwd(), "/content/drive/MyDrive/Reconnaissance_Facial_Supermarch-_V2_PDFs/ByteTrack/weights/", "v8_n.pt")

# Chemins et configurations
DATABASE_PATH = os.path.join(os.getcwd(), "data", "faces_db.pkl")
FACE_SAVE_FOLDER = os.path.join(os.getcwd(), "data", "saved_faces")
RECEIPTS_FOLDER = os.path.join(os.getcwd(), "data", "receipts")
DEFAULT_LOG_ACHAT = os.path.join(os.getcwd(), "data", "logs_personne_achat.txt")
DEFAULT_LOG_TRAITE = os.path.join(os.getcwd(), "data", "logs_personne_traité.txt")
RECEIPT_NUMBER_FILE = os.path.join(os.getcwd(), "data", "last_receipt_number.txt")

SIMILARITY_THRESHOLD = 0.45
DETECTION_SIZE = 640

# Paramètres pour Mediapipe FaceMesh
DESIRED_ROLL = -0.84
DESIRED_PITCH = 89.97
DESIRED_YAW = -91.44
ANGLE_TOLERANCE = 20.0
