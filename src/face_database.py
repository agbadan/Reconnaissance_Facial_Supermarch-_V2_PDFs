# src/face_database.py

import pickle
import os
import logging
from typing import Dict
import numpy as np

class FaceDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.faces_db = self.load_database()

    def load_database(self) -> Dict[str, np.ndarray]:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    db = pickle.load(f)
                logging.info("Base de données chargée avec succès.")
                return db
            except Exception as e:
                logging.error(f"Erreur lors du chargement de la base de données: {e}")
                return {}
        else:
            return {}

    def save(self) -> None:
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump(self.faces_db, f)
            logging.info("Base de données sauvegardée avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de la base de données: {e}")
