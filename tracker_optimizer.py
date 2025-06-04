# tracker_optimizer.py
from collections import defaultdict

class TrackerOptimizer:
    def __init__(self, max_age=30, distance_threshold=50):
        self.max_age = max_age
        self.distance_threshold = distance_threshold
        self.tracks = defaultdict(lambda: {'pos': None, 'age': 0})
        self.next_id = 1

    def update(self, detections):
        current_ids = []
        
        # Atualiza idade de todos os tracks existentes
        for track_id in self.tracks:
            self.tracks[track_id]['age'] += 1

        # Processa novas detecções
        for det in detections:
            xmin, ymin, xmax, ymax = det
            current_pos = ((xmin + xmax) // 2, ymax)  # Centro inferior
            
            best_match = None
            min_distance = float('inf')
            
            # Procura o track mais próximo
            for track_id, track in self.tracks.items():
                if track['pos'] is None:
                    continue
                distance = ((current_pos[0] - track['pos'][0])**2 + 
                           (current_pos[1] - track['pos'][1])**2)**0.5
                if distance < min_distance and distance < self.distance_threshold:
                    min_distance = distance
                    best_match = track_id
            
            if best_match is not None:
                current_ids.append(best_match)
                self.tracks[best_match]['pos'] = current_pos
                self.tracks[best_match]['age'] = 0
            else:
                current_ids.append(self.next_id)
                self.tracks[self.next_id] = {'pos': current_pos, 'age': 0}
                self.next_id += 1

        # Remove tracks antigos
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        return current_ids