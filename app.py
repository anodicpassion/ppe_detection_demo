from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import time
import json
from collections import deque
import threading
import base64
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        track_id = self.next_id
        self.tracks[track_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'disappeared': 0
        }
        self.next_id += 1
        return track_id

    def deregister(self, track_id):
        if track_id in self.tracks:
            del self.tracks[track_id]

    def update(self, bboxes):
        if not bboxes:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    self.deregister(track_id)
            return []

        centroids = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in bboxes]

        if not self.tracks:
            tracked_objects = [{'id': self.register(c, b), 'bbox': b} for c, b in zip(centroids, bboxes)]
            return tracked_objects

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]['centroid'] for tid in track_ids]
        distances = np.sqrt(((np.array(track_centroids)[:, None] - np.array(centroids)) ** 2).sum(axis=2))

        if distances.size == 0:
            for track_id in track_ids:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    self.deregister(track_id)
            return []

        rows, cols = linear_sum_assignment(distances)
        used_rows, used_cols = set(), set()
        tracked_objects = []

        for row, col in zip(rows, cols):
            if distances[row, col] < self.max_distance:
                track_id = track_ids[row]
                self.tracks[track_id].update({
                    'centroid': centroids[col],
                    'bbox': bboxes[col],
                    'disappeared': 0
                })
                tracked_objects.append({'id': track_id, 'bbox': bboxes[col]})
                used_rows.add(row)
                used_cols.add(col)

        for col, (centroid, bbox) in enumerate(zip(centroids, bboxes)):
            if col not in used_cols:
                track_id = self.register(centroid, bbox)
                tracked_objects.append({'id': track_id, 'bbox': bbox})

        for row, track_id in enumerate(track_ids):
            if row not in used_rows:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    self.deregister(track_id)

        return tracked_objects

class SafetyMonitor:
    def __init__(self, model_path="best.pt", conf_threshold=0.25, iou_threshold=0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_tracker = CentroidTracker(max_disappeared=30, max_distance=100)
        self.expected_classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                                'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
        self.class_names = self.model.names
        self.violations = deque(maxlen=100)
        self.prev_ppe_detections = {}
        self.ppe_history = {}
        self.load_status_icons()
        self.running = False
        self.paused = False
        self.cap = None
        self.thread = None
        self.frame_idx = 0
        self.fps = 0
        self.total_frames = -1

    def validate_class_mapping(self):
        model_classes = list(self.class_names.values()) if isinstance(self.class_names, dict) else self.class_names
        missing_classes = set(self.expected_classes) - set(model_classes)
        if missing_classes:
            print(f"⚠️ Warning: Expected classes not found in model: {missing_classes}")
        extra_classes = set(model_classes) - set(self.expected_classes)
        if extra_classes:
            print(f"ℹ️ Additional classes found in model: {extra_classes}")

    def load_status_icons(self):
        try:
            icon_size = (50, 50)
            icon_files = {
                'green_helmet': 'static/files/greenHelmet.png',
                'red_helmet': 'static/files/redHelmet.png',
                'green_mask': 'static/files/greenMask.png',
                'red_mask': 'static/files/redMask.png',
                'green_vest': 'static/files/greenVest.png',
                'red_vest': 'static/files/redVest.png'
            }
            for icon_name, file_path in icon_files.items():
                icon = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon = cv2.resize(icon, icon_size)
                    setattr(self, icon_name, icon)
                else:
                    print(f"⚠️ Warning: Could not load {icon_name} from {file_path}")
                    setattr(self, icon_name, None)
        except Exception as e:
            print(f"❌ Error loading status icons: {e}")
            for icon_name in icon_files:
                setattr(self, icon_name, None)

    def overlay_icon(self, frame, icon, x, y, fallback_text, color):
        if icon is not None and x >= 0 and y >= 0 and x + icon.shape[1] <= frame.shape[1] and y + icon.shape[0] <= frame.shape[0]:
            if len(icon.shape) == 3 and icon.shape[2] == 4:
                alpha = icon[:, :, 3] / 255.0
                for c in range(3):
                    frame[y:y+icon.shape[0], x:x+icon.shape[1], c] = (
                        alpha * icon[:, :, c] + (1 - alpha) * frame[y:y+icon.shape[0], x:x+icon.shape[1], c]
                    )
            else:
                if len(icon.shape) == 3:
                    gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
                else:
                    gray = icon
                mask = gray < 240
                frame[y:y+icon.shape[0], x:x+icon.shape[1]][mask] = icon[mask]
        else:
            cv2.putText(frame, fallback_text, (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def check_detection_consistency(self, ppe_detections, frame_idx):
        for ppe_type in ['hardhat', 'mask', 'safety_vest']:
            pos_key = f'{ppe_type}_positive'
            neg_key = f'{ppe_type}_negative'
            curr_pos = len(ppe_detections.get(pos_key, []))
            curr_neg = len(ppe_detections.get(neg_key, []))
            prev_pos = self.prev_ppe_detections.get(pos_key, 0)
            prev_neg = self.prev_ppe_detections.get(neg_key, 0)
            if frame_idx > 0 and (abs(curr_pos - prev_pos) > 3 or abs(curr_neg - prev_neg) > 3):
                print(f"⚠️ Detection inconsistency at frame {frame_idx}: {ppe_type} changed from {prev_pos}/{prev_neg} to {curr_pos}/{curr_neg}")
            self.prev_ppe_detections[pos_key] = curr_pos
            self.prev_ppe_detections[neg_key] = curr_neg

    def classify_detection(self, class_name):
        class_name_lower = class_name.lower().strip() if class_name else ''
        if class_name_lower == 'person':
            return {'category': 'person', 'is_person': True, 'is_ppe': False, 'is_negative': False}
        if any(term in class_name_lower for term in ['hardhat', 'hard hat', 'helmet']):
            if 'no' not in class_name_lower and '-' not in class_name_lower:
                return {'category': 'ppe_positive', 'is_person': False, 'is_ppe': True, 'is_negative': False, 'ppe_type': 'hardhat'}
        if any(term in class_name_lower for term in ['mask', 'face mask']):
            if 'no' not in class_name_lower and '-' not in class_name_lower:
                return {'category': 'ppe_positive', 'is_person': False, 'is_ppe': True, 'is_negative': False, 'ppe_type': 'mask'}
        if any(term in class_name_lower for term in ['safety vest', 'vest', 'hi-vis', 'high-vis']):
            if 'no' not in class_name_lower and '-' not in class_name_lower:
                return {'category': 'ppe_positive', 'is_person': False, 'is_ppe': True, 'is_negative': False, 'ppe_type': 'safety_vest'}
        if any(term in class_name_lower for term in ['no-hardhat', 'no hardhat', 'no-helmet', 'no helmet']):
            return {'category': 'ppe_negative', 'is_person': False, 'is_ppe': True, 'is_negative': True, 'ppe_type': 'hardhat'}
        if any(term in class_name_lower for term in ['no-mask', 'no mask', 'no-face mask']):
            return {'category': 'ppe_negative', 'is_person': False, 'is_ppe': True, 'is_negative': True, 'ppe_type': 'mask'}
        if any(term in class_name_lower for term in ['no-safety vest', 'no safety vest', 'no-vest', 'no vest']):
            return {'category': 'ppe_negative', 'is_person': False, 'is_ppe': True, 'is_negative': True, 'ppe_type': 'safety_vest'}
        if 'safety cone' in class_name_lower or 'cone' in class_name_lower:
            return {'category': 'safety_equipment', 'is_person': False, 'is_ppe': False, 'is_negative': False}
        if class_name_lower in ['machinery', 'vehicle']:
            return {'category': 'hazard', 'is_person': False, 'is_ppe': False, 'is_negative': False, 'hazard_type': class_name_lower}
        return {'category': 'other', 'is_person': False, 'is_ppe': False, 'is_negative': False}

    def check_ppe_compliance(self, person_id, person_bbox, ppe_detections, frame_idx):
        px1, py1, px2, py2 = person_bbox
        compliance = {
            'hardhat': False,
            'mask': False,
            'safety_vest': False,
            'violations': []
        }
        if person_id not in self.ppe_history:
            self.ppe_history[person_id] = deque(maxlen=10)
        for ppe_type in ['hardhat', 'mask', 'safety_vest']:
            positive_detections = ppe_detections.get(f'{ppe_type}_positive', [])
            negative_detections = ppe_detections.get(f'{ppe_type}_negative', [])
            explicit_violation = False
            for ppe_bbox in negative_detections:
                if self.bbox_overlap_check(person_bbox, ppe_bbox, ppe_type):
                    compliance[ppe_type] = False
                    explicit_violation = True
                    break
            if not explicit_violation:
                for ppe_bbox in positive_detections:
                    if self.bbox_overlap_check(person_bbox, ppe_bbox, ppe_type):
                        compliance[ppe_type] = True
                        break
            self.ppe_history[person_id].append(compliance[ppe_type])
            if len(self.ppe_history[person_id]) >= 3:
                recent_states = list(self.ppe_history[person_id])[-3:]
                if all(not state for state in recent_states) and not compliance[ppe_type]:
                    violation_name = f'No {ppe_type.replace("_", " ").title()}'
                    if violation_name not in compliance['violations']:
                        compliance['violations'].append(violation_name)
        return compliance

    def bbox_overlap_check(self, person_bbox, ppe_bbox, ppe_type):
        px1, py1, px2, py2 = person_bbox
        ppx1, ppy1, ppx2, ppy2 = ppe_bbox
        ppe_center_x = (ppx1 + ppx2) / 2
        ppe_center_y = (ppy1 + ppy2) / 2
        person_width = px2 - px1
        person_height = py2 - py1
        region_match = False
        if ppe_type == 'hardhat':
            head_region_bottom = py1 + (person_height * 0.45)
            region_match = (px1 <= ppe_center_x <= px2 and py1 <= ppe_center_y <= head_region_bottom)
        elif ppe_type == 'mask':
            face_region_bottom = py1 + (person_height * 0.55)
            region_match = (px1 <= ppe_center_x <= px2 and py1 <= ppe_center_y <= face_region_bottom)
        elif ppe_type == 'safety_vest':
            torso_top = py1 + (person_height * 0.15)
            torso_bottom = py1 + (person_height * 0.95)
            horizontal_tolerance = person_width * 0.2
            region_match = (px1 - horizontal_tolerance <= ppe_center_x <= px2 + horizontal_tolerance and 
                            torso_top <= ppe_center_y <= torso_bottom)
        if region_match:
            return True
        overlap_x = max(0, min(px2, ppx2) - max(px1, ppx1))
        overlap_y = max(0, min(py2, ppy2) - max(py1, ppy1))
        overlap_area = overlap_x * overlap_y
        if overlap_area > 0:
            person_area = (px2 - px1) * (py2 - py1)
            ppe_area = (ppx2 - ppx1) * (ppy2 - ppy1)
            overlap_ratio_person = overlap_area / person_area if person_area > 0 else 0
            overlap_ratio_ppe = overlap_area / ppe_area if ppe_area > 0 else 0
            threshold = 0.15 if ppe_type == 'safety_vest' else 0.25
            if overlap_ratio_ppe > threshold or overlap_ratio_person > threshold:
                return True
        return False

    def adjust_icon_position(self, frame, x, y, icon_size, occupied_positions):
        icon_w, icon_h = icon_size
        for ox, oy in occupied_positions:
            if abs(x - ox) < icon_w and abs(y - oy) < icon_h:
                x = ox + icon_w + 5
                if x + icon_w > frame.shape[1]:
                    x = ox - icon_w - 5
                if x < 0 or x + icon_w > frame.shape[1]:
                    y += icon_h + 5
                    x = ox
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + icon_w > frame.shape[1]:
            x = frame.shape[1] - icon_w
        if y + icon_h > frame.shape[0]:
            y = frame.shape[0] - icon_h
        return x, y

    def process_frame(self, frame, frame_idx):
        start_time = time.time()
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
        result = results[0]
        annotated_frame = frame.copy()
        person_bboxes = []
        ppe_detections = {
            'hardhat_positive': [],
            'hardhat_negative': [],
            'mask_positive': [],
            'mask_negative': [],
            'safety_vest_positive': [],
            'safety_vest_negative': []
        }
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    x1, y1, x2, y2 = map(int, box)
                    bbox = [x1, y1, x2, y2]
                    classification = self.classify_detection(class_name)
                    if classification['is_person']:
                        person_bboxes.append(bbox)
                    elif classification['is_ppe']:
                        ppe_type = classification['ppe_type']
                        key = f"{ppe_type}_negative" if classification['is_negative'] else f"{ppe_type}_positive"
                        if key in ppe_detections:
                            ppe_detections[key].append(bbox)
        self.check_detection_consistency(ppe_detections, frame_idx)
        tracked_persons = self.person_tracker.update(person_bboxes)
        person_compliance = {}
        violations = []
        for person in tracked_persons:
            person_id = person['id']
            person_bbox = person['bbox']
            compliance = self.check_ppe_compliance(person_id, person_bbox, ppe_detections, frame_idx)
            person_compliance[person_id] = compliance
            if compliance['violations']:
                violation = {
                    'frame': frame_idx,
                    'person_id': person_id,
                    'violations': compliance['violations'],
                    'bbox': person_bbox
                }
                violations.append(violation)
                self.violations.append(violation)
        self.draw_annotations(annotated_frame, tracked_persons, person_compliance)
        processing_time = time.time() - start_time
        self.fps = 1.0 / processing_time if processing_time > 0 else 0
        stats = {
            'total_persons': len(tracked_persons),
            'safe_persons': sum(1 for pc in person_compliance.values() if not pc['violations']),
            'unsafe_persons': sum(1 for pc in person_compliance.values() if pc['violations']),
            'helmet_safe': sum(1 for pc in person_compliance.values() if pc['hardhat']),
            'helmet_unsafe': sum(1 for pc in person_compliance.values() if 'No Hardhat' in pc['violations']),
            'vest_safe': sum(1 for pc in person_compliance.values() if pc['safety_vest']),
            'vest_unsafe': sum(1 for pc in person_compliance.values() if 'No Safety Vest' in pc['violations']),
            'mask_safe': sum(1 for pc in person_compliance.values() if pc['mask']),
            'mask_unsafe': sum(1 for pc in person_compliance.values() if 'No Mask' in pc['violations']),
            'fps': self.fps,
            'progress': (frame_idx / self.total_frames * 100) if self.total_frames > 0 else 0
        }
        return annotated_frame, tracked_persons, ppe_detections, violations, person_compliance, stats

    def draw_annotations(self, frame, tracked_persons, person_compliance):
        occupied_positions = []
        for person in tracked_persons:
            x1, y1, x2, y2 = map(int, person['bbox'])
            track_id = person['id']
            compliance = person_compliance.get(track_id, {})
            violations = compliance.get('violations', [])
            color = (0, 255, 0) if not violations else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"ID-{track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + label_size[0] + 10, y1 - 5), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            icon_x_base = x1 - 55
            icon_y_start = y1 - 135
            icon_spacing = 45
            icon_size = (50, 50)
            for ppe_type, icon, red_icon, text, text_color in [
                ('hardhat', self.green_helmet, self.red_helmet, 'H', (0, 255, 0) if compliance.get('hardhat', False) else (0, 0, 255)),
                ('mask', self.green_mask, self.red_mask, 'M', (0, 255, 0) if compliance.get('mask', False) else (0, 0, 255)),
                ('safety_vest', self.green_vest, self.red_vest, 'V', (0, 255, 0) if compliance.get('safety_vest', False) else (0, 0, 255))
            ]:
                icon_y = icon_y_start + icon_spacing * ['hardhat', 'mask', 'safety_vest'].index(ppe_type)
                icon_x, icon_y = self.adjust_icon_position(frame, icon_x_base, icon_y, icon_size, occupied_positions)
                icon_to_use = icon if compliance.get(ppe_type, False) else red_icon
                self.overlay_icon(frame, icon_to_use, icon_x, icon_y, text, text_color)
                occupied_positions.append((icon_x, icon_y))

    def process_video(self, input_path):
        try:
            if input_path == 0 or str(input_path).lower() == 'webcam':
                self.cap = cv2.VideoCapture(0)
                self.total_frames = -1
            else:
                self.cap = cv2.VideoCapture(input_path)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {input_path}")
            self.frame_idx = 0
            while self.running and self.cap.isOpened():
                if self.paused:
                    time.sleep(0.1)
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    if input_path != 0:
                        break
                    continue
                annotated_frame, tracked_persons, ppe_detections, violations, person_compliance, stats = self.process_frame(frame, self.frame_idx)
                _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('frame', {'image': frame_base64})
                socketio.emit('stats', stats)
                for violation in violations:
                    socketio.emit('violation', violation)
                self.frame_idx += 1
        except Exception as e:
            print(f"❌ Error processing video: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.running = False

    def start(self, input_path, conf_threshold, iou_threshold):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self.process_video, args=(input_path,))
        self.thread.start()

    def pause(self):
        self.paused = True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        self.cap = None
        self.frame_idx = 0
        self.fps = 0
        self.total_frames = -1

monitor = SafetyMonitor(model_path='best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'success': True, 'file_path': file_path})
    return jsonify({'success': False, 'error': 'File upload failed'})

@socketio.on('start_detection')
def handle_start_detection(data):
    input_path = data.get('input_path', 0)  # Default to webcam
    conf = float(data.get('conf', 0.25))
    iou = float(data.get('iou', 0.45))
    monitor.start(input_path, conf, iou)

@socketio.on('pause_detection')
def handle_pause_detection():
    monitor.pause()

@socketio.on('stop_detection')
def handle_stop_detection():
    monitor.stop()

@socketio.on('update_settings')
def handle_update_settings(data):
    monitor.conf_threshold = float(data.get('conf', monitor.conf_threshold))
    monitor.iou_threshold = float(data.get('iou', monitor.iou_threshold))

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5500)