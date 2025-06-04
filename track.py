import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# Parâmetros de filtro/calibração
MIN_AREA = 1000           # área mínima da bbox (ex: 40x30)
MIN_LIFE_FRAMES = 5       # frames mínimos para contar o ID
MIN_CONF = 0.7           # confiança mínima da detecção
MIN_ASPECT = 0.25          # proporção mínima (w/h)
MAX_ASPECT = 1.4          # proporção máxima (w/h)

# DeepSORT mais robusto
tracker = DeepSort(max_age=90, n_init=5, max_iou_distance=0.5, nms_max_overlap=1.0)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video.mp4")
ymax_threshold = 400

unique_close_ids = set()
unique_far_ids = set()
id_life = {}  # track_id: número de frames ativo

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    person_boxes = []
    vehicle_boxes = []

    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0]) if hasattr(box.cls, '__getitem__') else int(box.cls)
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            width = xmax - xmin
            height = ymax - ymin
            area = width * height
            aspect = width / (height + 1e-6)
            if cls == 0 and conf > MIN_CONF and area > MIN_AREA and MIN_ASPECT < aspect < MAX_ASPECT:
                person_boxes.append((xmin, ymin, xmax, ymax, conf))
            elif cls in [1, 3]:  # bicicleta ou moto
                vehicle_boxes.append((xmin, ymin, xmax, ymax))

        # Filtrar pessoas que NÃO estão sobre motos/bicicletas
        filtered_person_boxes = []
        for pbox in person_boxes:
            is_on_vehicle = False
            for vbox in vehicle_boxes:
                if iou(pbox, vbox) > 0.2:
                    is_on_vehicle = True
                    break
            if not is_on_vehicle:
                filtered_person_boxes.append(pbox)

        # Converter para formato DeepSORT: ([xmin, ymin, width, height], conf, class_id)
        deepsort_detections = []
        for pbox in filtered_person_boxes:
            xmin, ymin, xmax, ymax, conf = pbox
            width = xmax - xmin
            height = ymax - ymin
            area = width * height
            aspect = width / (height + 1e-6)
            if area < MIN_AREA or not (MIN_ASPECT < aspect < MAX_ASPECT):
                continue  # Ignora detecção pequena ou desproporcional
            deepsort_detections.append(([xmin, ymin, width, height], conf, 0))

        if deepsort_detections:
            tracks = tracker.update_tracks(deepsort_detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                width = r - l
                height = b - t
                area = width * height
                aspect = width / (height + 1e-6)
                if area < MIN_AREA or not (MIN_ASPECT < aspect < MAX_ASPECT):
                    continue  # Ignora detecção pequena ou desproporcional

                # Atualiza vida do ID
                id_life[track_id] = id_life.get(track_id, 0) + 1
                if id_life[track_id] < MIN_LIFE_FRAMES:
                    continue  # Só conta IDs que viveram o suficiente

                # Classifique por posição
                if b >= ymax_threshold:
                    color = (0, 255, 0)
                    unique_close_ids.add(track_id)
                else:
                    color = (0, 0, 255)
                    unique_far_ids.add(track_id)

                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Exibir as contagens únicas em tempo real
    cv2.putText(frame, f"Total Perto: {len(unique_close_ids)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Longe: {len(unique_far_ids)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f"Total: {len(unique_close_ids.union(unique_far_ids))}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Calcule os valores finais 
final_close = len(unique_close_ids)
final_far = len(unique_far_ids)
final_total = len(unique_close_ids.union(unique_far_ids))

# Crie a string do resultado
resultado = (
    f"Total Perto: {final_close}\n"
    f"Total Longe: {final_far}\n"
    f"Total Geral: {final_total}\n"
)

# Salvar em arquivo .txt
with open("resultado_contagem.txt", "w") as f:
    f.write(resultado)