import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import mediapipe as mp

# --- Configuration ---
MODEL_PATH = 'isl_model.h5'
LABELS_PATH = 'labels.npy'
IMG_HEIGHT = 64
IMG_WIDTH = 64
ROLLING_WINDOW = 10
SMOOTH_ALPHA = 0.3  # smoothing factor for bounding box

# --- Load Model and Labels ---
if not (tf.io.gfile.exists(MODEL_PATH) and tf.io.gfile.exists(LABELS_PATH)):
    print("Error: Model or labels not found. Train first.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
class_labels = np.load(LABELS_PATH)
print("Model and labels loaded successfully.\n")

# --- Mediapipe Hand Detection ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False,
                                max_num_hands=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

# --- Helper Functions ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    normalized = resized / 255.0
    return normalized.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

def predict_frame(img):
    try:
        pred = model.predict(preprocess_image(img), verbose=0)[0]
        top3_idx = pred.argsort()[-3:][::-1]
        top3 = [(class_labels[i], pred[i]*100) for i in top3_idx]
        conf = top3[0][1]
        if conf < 50:
            return "Prediction uncertain", (0,255,255), conf
        text = " | ".join([f"{lbl}:{c:.1f}%" for lbl,c in top3])
        return text, (0,255,0), conf
    except:
        return "Error", (0,0,255), 0

def get_combined_roi(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    x_min = w; x_max = 0; y_min = h; y_max = 0
    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
    pad = 20
    x_min, y_min = max(0, x_min-pad), max(0, y_min-pad)
    x_max, y_max = min(w, x_max+pad), min(h, y_max+pad)
    roi = frame[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)

# --- Black & White Menu ---
menu_img = np.zeros((400,500,3), dtype=np.uint8)
labels = ["Live Prediction", "Capture Once"]
button_w, button_h = 300, 60
start_y, gap = 80, 40
button_coords = []

for i, label in enumerate(labels):
    top_left = (100, start_y + i*(button_h+gap))
    bottom_right = (100+button_w, start_y + i*(button_h+gap)+button_h)
    cv2.rectangle(menu_img, top_left, bottom_right, (255,255,255), -1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = top_left[0] + (button_w - text_size[0]) // 2
    text_y = top_left[1] + (button_h + text_size[1]) // 2
    cv2.putText(menu_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    button_coords.append((top_left, bottom_right))

mode_selected = None
def menu_click(event, x, y, flags, param):
    global mode_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (top_left, bottom_right) in enumerate(button_coords):
            if top_left[0]<x<bottom_right[0] and top_left[1]<y<bottom_right[1]:
                mode_selected = "live" if i==0 else "capture"

cv2.namedWindow("ISL Menu")
cv2.setMouseCallback("ISL Menu", menu_click)
print("Click a button or press L=Live, C=Capture")

while mode_selected is None:
    cv2.imshow("ISL Menu", menu_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('l'):
        mode_selected = "live"
    elif key == ord('c'):
        mode_selected = "capture"

cv2.destroyAllWindows()
print(f"Selected mode: {mode_selected}")

# --- Open Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# --- Mode: Live Prediction ---
if mode_selected=="live":
    cv2.namedWindow("ISL Live Prediction")
    pred_queue = deque(maxlen=ROLLING_WINDOW)
    prev_time = time.time()
    smoothed_bbox = None
    print("Live Prediction started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        if results.multi_hand_landmarks:
            roi, bbox = get_combined_roi(frame, results.multi_hand_landmarks)
            if smoothed_bbox is None:
                smoothed_bbox = bbox
            else:
                smoothed_bbox = tuple(int(SMOOTH_ALPHA*b + (1-SMOOTH_ALPHA)*s) for b,s in zip(bbox, smoothed_bbox))
            x_min, y_min, x_max, y_max = smoothed_bbox
            roi = frame[y_min:y_max, x_min:x_max]

            # Prediction
            pred_queue.append(model.predict(preprocess_image(roi), verbose=0)[0])
            avg_pred = np.mean(pred_queue, axis=0)
            top3_idx = avg_pred.argsort()[-3:][::-1]
            top3 = [(class_labels[i], avg_pred[i]*100) for i in top3_idx]
            conf = top3[0][1]

            if conf>80: color=(0,255,0)
            elif conf>50: color=(0,255,255)
            else: color=(0,0,255)
            text = " | ".join([f"{lbl}:{c:.1f}%" for lbl,c in top3])

            # Draw
            cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),color,2)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            text = "No hand detected"
            color=(0,0,255)
            smoothed_bbox = None

        # FPS
        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time

        # Display
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
        cv2.putText(frame,f"FPS: {fps:.1f}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        cv2.putText(frame,"Q=Quit",(10,130),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)
        cv2.imshow("ISL Live Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- Mode: Capture Once ---
elif mode_selected=="capture":
    cv2.namedWindow("Capture Once")
    print("Capture Once started. Aim hand at camera. Press SPACE to predict, Q to quit.")

    capture_text = ""
    last_pred_time = 0
    smoothed_bbox = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Instructions
        cv2.putText(frame,"SPACE=Predict | Q=Quit",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):   # Predict on SPACE
            if results.multi_hand_landmarks:
                roi, bbox = get_combined_roi(frame, results.multi_hand_landmarks)
                if smoothed_bbox is None:
                    smoothed_bbox = bbox
                else:
                    smoothed_bbox = tuple(int(SMOOTH_ALPHA*b + (1-SMOOTH_ALPHA)*s) for b,s in zip(bbox, smoothed_bbox))
                x_min, y_min, x_max, y_max = smoothed_bbox
                roi = frame[y_min:y_max, x_min:x_max]
                text, color, conf = predict_frame(roi)
                cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),color,2)
            else:
                text, color, conf = "No hand detected", (0,0,255), 0
            capture_text = text
            last_pred_time = time.time()

        # Auto-hide after 3 seconds
        if time.time() - last_pred_time > 3:
            capture_text = ""

        # Display
        if capture_text:
            cv2.putText(frame, capture_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Capture Once", frame)
        if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()
