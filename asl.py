import cv2
import numpy as np
import tensorflow as tf
import time
from collections import Counter

# ── 1. Load Model ─────────────────────────────────────────────
MODEL_PATH = "asl.keras"

model = tf.keras.models.load_model(MODEL_PATH)
_, height, width, channels = model.input_shape
print(f"✅ Model loaded! Input: {height}x{width}x{channels}")

# ── 2. Classes (same order as training) ───────────────────────
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# ── 3. Parameters ─────────────────────────────────────────────
CONF_THRESHOLD  = 0.85
PREDICT_EVERY   = 3
BUFFER_SIZE     = 8
MIN_VOTE_COUNT  = 5
HOLD_TIME       = 1.0

# ── 4. Pre-allocate buffer ────────────────────────────────────
input_buffer = np.empty((1, height, width, 3), dtype=np.float32)

def preprocess(roi):
    """
    Match training data preprocessing:
    - Convert to RGB (training images were RGB)
    - Resize to model input size
    - Normalize 0-1
    - Apply CLAHE to fix lighting differences
    """
    # Convert BGR → RGB (OpenCV reads BGR, model trained on RGB)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # CLAHE on L channel to normalize lighting
    roi_lab  = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_lab[:, :, 0] = clahe.apply(roi_lab[:, :, 0])
    roi_rgb  = cv2.cvtColor(roi_lab, cv2.COLOR_LAB2RGB)

    # Resize & normalize
    resized  = cv2.resize(roi_rgb, (width, height))
    np.divide(resized, 255.0, out=input_buffer[0])
    return input_buffer

# ── 5. Warm up ────────────────────────────────────────────────
print("Warming up...")
model(input_buffer, training=False)
print("Ready! Show your hand inside the green box.")

# ── 6. Sentence state ─────────────────────────────────────────
current_word    = ""
sentence        = ""
last_appended   = ""
hold_start_time = None
stable_label    = ""
prediction_buffer = []

# ── 7. Webcam ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
label       = "..."
confidence  = 0.0
prev_time   = time.time()

print("Controls: q=quit  c=clear  b=backspace")

# ── 8. Text wrap helper ───────────────────────────────────────
def draw_wrapped(frame, text, x, y, max_w, scale, color, thick):
    line   = ""
    line_y = y
    for ch in text:
        test = line + ch
        (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw > max_w and line:
            cv2.putText(frame, line, (x, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
            line   = ch
            line_y += 28
        else:
            line = test
    if line:
        cv2.putText(frame, line, (x, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

# ── 9. Main Loop ──────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ── ROI: square box, upper center, no face ────────────────
    box_size = int(min(w, h) * 0.5)
    cx       = w // 2
    x1       = cx - box_size // 2
    x2       = cx + box_size // 2
    y1       = 20
    y2       = y1 + box_size
    roi      = frame[y1:y2, x1:x2]

    # ── Predict ───────────────────────────────────────────────
    if frame_count % PREDICT_EVERY == 0:
        inp        = preprocess(roi)
        preds      = model(inp, training=False).numpy()[0]
        class_id   = int(np.argmax(preds))
        confidence = float(preds[class_id])
        raw_label  = classes[class_id]

        prediction_buffer.append(raw_label)
        if len(prediction_buffer) > BUFFER_SIZE:
            prediction_buffer.pop(0)

        most_common, count = Counter(prediction_buffer).most_common(1)[0]

        if count >= MIN_VOTE_COUNT and confidence > CONF_THRESHOLD:
            label = most_common
        else:
            label = "..."

    # ── Hold Timer & Append ───────────────────────────────────
    now = time.time()

    if label not in ("...",):
        if label != stable_label:
            stable_label    = label
            hold_start_time = now
        else:
            held = now - hold_start_time
            if held >= HOLD_TIME and label != last_appended:
                if label == "space":
                    if current_word:
                        sentence     += current_word + " "
                        current_word  = ""
                elif label == "del":
                    if current_word:
                        current_word = current_word[:-1]
                    elif sentence:
                        sentence = sentence[:-1]
                elif label == "nothing":
                    pass
                else:
                    current_word += label.upper()

                last_appended   = label
                hold_start_time = now   # reset for next char
    else:
        stable_label    = ""
        hold_start_time = None
        last_appended   = ""

    # ── FPS ───────────────────────────────────────────────────
    curr_time   = time.time()
    fps_display = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time   = curr_time

    # ── Draw ROI box ──────────────────────────────────────────
    if label == "...":
        box_color = (0, 255, 255)      # yellow = unstable
    elif confidence > CONF_THRESHOLD:
        box_color = (0, 255, 0)        # green  = confident
    else:
        box_color = (0, 0, 255)        # red    = low conf

    # Outer guide box (where to place hand)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Corner markers to make ROI obvious
    corner_len = 20
    for (cx_, cy_, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx_, cy_), (cx_ + dx*corner_len, cy_), (255,255,255), 3)
        cv2.line(frame, (cx_, cy_), (cx_, cy_ + dy*corner_len), (255,255,255), 3)

    # Hold progress bar at bottom of ROI
    if label not in ("...",) and hold_start_time is not None:
        held     = min(now - hold_start_time, HOLD_TIME)
        progress = int((held / HOLD_TIME) * (x2 - x1))
        cv2.rectangle(frame, (x1, y2 - 8), (x1 + progress, y2), (0, 255, 0), -1)
        cv2.rectangle(frame, (x1, y2 - 8), (x2, y2), (100, 100, 100), 1)

    # Prediction text above box
    pred_text = f"{label}  ({confidence*100:.1f}%)" if label != "..." else "..."
    cv2.putText(frame, pred_text, (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2)

    # FPS top left
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ── Bottom Panel ──────────────────────────────────────────
    panel_y = int(h * 0.78)
    cv2.rectangle(frame, (0, panel_y), (w, h), (30, 30, 30), -1)

    # Word
    cv2.putText(frame, "Word:", (10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, current_word if current_word else "_",
                (75, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 150), 2)

    # Sentence
    cv2.putText(frame, "Sent:", (10, panel_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    full = sentence + current_word
    draw_wrapped(frame, full if full else "_",
                 75, panel_y + 55, w - 85, 0.7, (255, 255, 255), 2)

    # Controls hint
    cv2.putText(frame, "q:quit  c:clear  b:backspace  nothing:pause",
                (10, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    cv2.imshow("ASL Sign → Sentence", frame)

    # ── Keys ──────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
        current_word = ""
        last_appended = ""
        stable_label = ""
        hold_start_time = None
        prediction_buffer.clear()
    elif key == ord('b'):
        if current_word:
            current_word = current_word[:-1]
        elif sentence:
            sentence = sentence.rstrip()
            if sentence:
                sentence = sentence[:-1] + " "

cap.release()
cv2.destroyAllWindows()
print(f"\n📝 Final sentence: {sentence + current_word}")