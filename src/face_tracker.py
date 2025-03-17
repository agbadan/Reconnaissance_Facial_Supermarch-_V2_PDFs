# src/face_tracker.py

import numpy as np
import logging
from typing import Any, Dict, List, Tuple

class FaceTracker:
    def __init__(self, similarity_threshold: float = 0.45):
        self.similarity_threshold = similarity_threshold
        self.tracked_faces: Dict[str, Dict[str, Any]] = {}
        self.bt_id_to_label: Dict[int, str] = {}

    def update(self, detected_faces: List[Any], bt_id: int, face_db: Dict[str, np.ndarray],
               label_input: str, frame_count: int) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        logs: List[str] = []
        for face_obj in detected_faces:
            # Si le visage a déjà été traité
            if bt_id in self.bt_id_to_label:
                label_assigned = self.bt_id_to_label[bt_id]
                logs.append(f"Frame {frame_count} : Visage déjà connu ({label_assigned}). Mise à jour du suivi.")
                data = self.tracked_faces[label_assigned]
                new_count = data['count'] + 1
                new_embedding = (data['embedding'] * data['count'] + face_obj.normed_embedding) / new_count
                self.tracked_faces[label_assigned] = {'embedding': new_embedding, 'count': new_count, 'bbox': face_obj.bbox}
            else:
                current_embedding = face_obj.normed_embedding
                best_similarity_db = 0.0
                best_match_db = None
                # Vérification dans la base de données
                for stored_label, stored_embedding in face_db.items():
                    similarity = np.dot(current_embedding, stored_embedding)
                    if similarity > best_similarity_db:
                        best_similarity_db = similarity
                        best_match_db = stored_label
                if best_similarity_db >= self.similarity_threshold:
                    label_assigned = best_match_db
                    logs.append(f"Frame {frame_count} : Visage reconnu (base de données) : {label_assigned} (similarité: {best_similarity_db:.2%}).")
                    if label_assigned in self.tracked_faces:
                        data = self.tracked_faces[label_assigned]
                        new_count = data['count'] + 1
                        new_embedding = (data['embedding'] * data['count'] + current_embedding) / new_count
                        self.tracked_faces[label_assigned] = {'embedding': new_embedding, 'count': new_count, 'bbox': face_obj.bbox}
                    else:
                        self.tracked_faces[label_assigned] = {'embedding': current_embedding, 'count': 1, 'bbox': face_obj.bbox}
                else:
                    # Vérification dans le suivi actuel pour détecter un visage similaire
                    best_similarity_tracked = 0.0
                    best_match_tracked = None
                    for tracker_label, data in self.tracked_faces.items():
                        similarity = np.dot(current_embedding, data['embedding'])
                        if similarity > best_similarity_tracked:
                            best_similarity_tracked = similarity
                            best_match_tracked = tracker_label
                    if best_similarity_tracked >= self.similarity_threshold:
                        label_assigned = best_match_tracked
                        logs.append(f"Frame {frame_count} : Visage similaire trouvé dans le suivi ({label_assigned}) (similarité: {best_similarity_tracked:.2%}).")
                        data = self.tracked_faces[label_assigned]
                        new_count = data['count'] + 1
                        new_embedding = (data['embedding'] * data['count'] + current_embedding) / new_count
                        self.tracked_faces[label_assigned] = {'embedding': new_embedding, 'count': new_count, 'bbox': face_obj.bbox}
                    else:
                        # Nouveau visage détecté
                        if label_input.strip() != "":
                            new_label = label_input.strip()
                        else:
                            indices = [int(k.split("_")[1]) for k in face_db.keys() if k.startswith("personne_") and k.split("_")[1].isdigit()]
                            max_index = max(indices) if indices else 0
                            new_label = f"personne_{max_index + 1}"
                        label_assigned = new_label
                        logs.append(f"Frame {frame_count} : Nouveau visage détecté et enregistré sous l'identifiant {new_label}.")
                        self.tracked_faces[label_assigned] = {'embedding': current_embedding, 'count': 1, 'bbox': face_obj.bbox}
                        face_db[new_label] = current_embedding
                self.bt_id_to_label[bt_id] = label_assigned
            face_obj.assigned_label = label_assigned
        return self.tracked_faces, logs
