"""
Naruto hand-sign data collection tool.

Shows your webcam with live MediaPipe hand tracking so you get instant feedback
on whether both hands are being detected. Cycle through the 12 seals, capture
normalized landmark vectors, and they're saved straight to disk as training-ready
data.

Controls (focus the video window first):
    n / p      next / previous label (the 12 seals + "none")
    SPACE      capture a single frame for the current label
    h          hold-to-capture: toggle continuous capture ON/OFF
                (captures every frame while you hold a pose and move slightly)
    d          delete the last captured sample for the current label
    s          show per-label sample counts in the console
    q          quit (data is saved continuously, nothing is lost on quit)

The "none" label (after bird) is the TRANSITION class: toggle hold-capture on
and weave continuously between the prompted seal pairs WITHOUT settling on
either one. A motion gate drops frames where your hands pause, so accidentally
settling on a real seal is not recorded as none. SPACE (single capture) is
ungated — use it for deliberate static garbage like half-formed seals.

Each captured sample is one row: 2 hands x 21 landmarks x 3 coords = 126 values,
plus the label. Missing hands are stored as zeros and flagged, so the occlusion
cases are preserved honestly rather than silently dropped.

Output: data/landmarks.csv  (appended to, so you can collect across many sessions)
"""

import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

# ----------------------------------------------------------------------------
# The 12 canonical seals. Order is fixed so label indices stay stable forever.
# Do NOT reorder this list once you've collected data, or your labels shift.
# ----------------------------------------------------------------------------
SIGNS = [
    "rat", "ox", "tiger", "hare", "dragon", "snake",
    "horse", "ram", "monkey", "boar", "dog", "bird",
]

# "none" = the transition class: hands in motion BETWEEN seals, half-formed
# shapes, garbage. Appended LAST so the 12 real sign indices never shift.
# The model learns transit-shapes -> none, which fails the web app's confidence
# gate instead of misfiring as a wrong sign mid-combo.
NONE_LABEL = "none"
LABELS = SIGNS + [NONE_LABEL]

# Transition pairs to prompt during none-capture, taken from the duel's attack
# chains (these exact A->B transits are where misfires happen in play).
TRANSIT_PAIRS = [
    ("tiger", "horse"), ("tiger", "dog"), ("dog", "ram"), ("tiger", "snake"),
    ("snake", "boar"), ("boar", "ox"), ("ox", "hare"), ("hare", "monkey"),
    ("dragon", "ram"), ("ram", "snake"), ("ram", "boar"), ("dog", "bird"),
    ("dog", "monkey"), ("monkey", "horse"), ("free weave", "any signs"),
]
PAIR_ROTATE_SEC = 6      # rotate the prompted pair every N seconds

# Motion gate for none-capture: only save a frame if the WRIST (landmark 0) is
# traveling. Measured in RAW image coordinates — not the normalized features,
# which are wrist-centered/scale-normalized and so contain no hand-travel
# signal at all, only finger jitter (which explodes when interlocked seals
# occlude one hand). The wrist is MediaPipe's most stable landmark and stays
# visible even when fingers are occluded: wrists planted = holding a seal,
# wrists traveling = in transit. Units: fraction of the frame per frame.
# CALIBRATE with the HUD: hold a seal (motion should sit clearly below this),
# weave (clearly above); set the threshold between your two readings.
# 0.01 = calibrated on yogi's setup: still-hand tremor/jitter reads below it,
# weaving above. Lower toward 0.007 if samples accrue too slowly mid-weave.
NONE_MIN_MOTION = 0.01

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

# 2 hands * 21 landmarks * 3 coords
N_LANDMARK_VALUES = 2 * 21 * 3

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def normalize_hand(landmarks):
    """
    Normalize a single hand's 21 landmarks to be translation- and scale-invariant.

    - Translate so the wrist (landmark 0) sits at the origin.
    - Scale by the distance from wrist to middle-finger MCP (landmark 9), which is
      a stable proxy for hand size regardless of distance from the camera.

    Returns a flat list of 63 floats (21 * 3).
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist  # center on wrist

    scale = np.linalg.norm(pts[9])  # wrist -> middle MCP distance
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale

    return pts.flatten().tolist()


def build_row(results):
    """
    Turn a MediaPipe result into a 126-value feature row.

    Slot 0 = Left hand, slot 1 = Right hand (by MediaPipe's handedness label).
    A missing hand is all zeros. Returns (row, n_hands_detected).
    """
    left = [0.0] * 63
    right = [0.0] * 63
    n_hands = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm_set, handed in zip(results.multi_hand_landmarks,
                                  results.multi_handedness):
            label = handed.classification[0].label  # "Left" or "Right"
            norm = normalize_hand(lm_set.landmark)
            if label == "Left":
                left = norm
            else:
                right = norm
            n_hands += 1

    return left + right, n_hands


def ensure_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        header = (
            [f"L{i}_{a}" for i in range(21) for a in ("x", "y", "z")]
            + [f"R{i}_{a}" for i in range(21) for a in ("x", "y", "z")]
            + ["n_hands", "label", "subject", "timestamp"]
        )
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(header)


def load_counts():
    counts = {s: 0 for s in LABELS}
    if not os.path.exists(CSV_PATH):
        return counts
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            label = row[-3]
            if label in counts:
                counts[label] += 1
    return counts


def append_sample(row, n_hands, label, subject):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(
            row + [n_hands, label, subject, datetime.now().isoformat()]
        )


def delete_last_for_label(label, subject):
    """Remove the most recent row matching label AND subject. Returns True if removed."""
    if not os.path.exists(CSV_PATH):
        return False
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    for i in range(len(body) - 1, -1, -1):
        # new schema tail: [..., n_hands, label, subject, timestamp]
        if body[i] and body[i][-3] == label and body[i][-2] == subject:
            del body[i]
            with open(CSV_PATH, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(body)
            return True
    return False


def main():
    ensure_csv()
    counts = load_counts()

    subject = input("Enter subject ID (e.g. your first name, lowercase, no spaces): ").strip().lower()
    while not subject or " " in subject:
        subject = input("Invalid. Use a single lowercase word (e.g. 'alex'): ").strip().lower()
    print(f"Collecting as subject: '{subject}'\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (device 0). "
              "Close other apps using the camera, or change the index in cv2.VideoCapture(0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sign_idx = 0
    hold_capture = False
    last_capture_time = 0.0
    HOLD_INTERVAL = 0.08  # seconds between captures while holding (~12/sec)
    prev_wrists = None    # {"Left"/"Right": (x, y)} raw image coords, for the none gate
    motion = 0.0

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print(__doc__)
    print("Starting. Focus the video window to use the keyboard.\n")

    dropped = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            dropped += 1
            if dropped == 30:
                print("\nStill no frames after 30 tries — another app is holding the "
                      "camera.\nUsual culprit: the browser tab running the recognizer "
                      "(its getUserMedia stream stays live).\nClose that tab (or "
                      "Zoom/Teams/OBS), then rerun this script.\n")
            else:
                print("Dropped a frame, retrying...")
            continue
        dropped = 0

        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        row, n_hands = build_row(results)

        sign = LABELS[sign_idx]
        is_none_mode = sign == NONE_LABEL

        # None-gate motion = wrist travel in RAW image coords (see the constant's
        # comment for why: normalized features carry no travel signal, and
        # finger landmarks jitter wildly under occlusion — the wrist does
        # neither). MIN across wrists seen in both frames: every visible hand
        # must be traveling, so a planted interlocked seal reads still no
        # matter what the occluded fingers appear to do. EMA-smoothed.
        wrists = {}
        if results.multi_hand_landmarks and results.multi_handedness:
            for lm_set, handed in zip(results.multi_hand_landmarks,
                                      results.multi_handedness):
                w = lm_set.landmark[0]
                wrists[handed.classification[0].label] = (w.x, w.y)
        if wrists and prev_wrists:
            deltas = [
                float(np.hypot(x - prev_wrists[k][0], y - prev_wrists[k][1]))
                for k, (x, y) in wrists.items() if k in prev_wrists
            ]
            raw_motion = min(deltas) if deltas else 0.0
            motion = 0.5 * motion + 0.5 * raw_motion
        else:
            motion = 0.0
        prev_wrists = wrists if wrists else None

        # draw landmarks
        if results.multi_hand_landmarks:
            for lm_set in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, lm_set, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # auto-capture while holding
        if hold_capture and time.time() - last_capture_time > HOLD_INTERVAL:
            if n_hands > 0 and (not is_none_mode or motion >= NONE_MIN_MOTION):
                append_sample(row, n_hands, sign, subject)
                counts[sign] += 1
                last_capture_time = time.time()

        # ---- HUD ----
        hud_color = (0, 255, 0) if n_hands == 2 else (
            (0, 200, 255) if n_hands == 1 else (0, 0, 255))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        if is_none_mode:
            a, b = TRANSIT_PAIRS[int(time.time() / PAIR_ROTATE_SEC) % len(TRANSIT_PAIRS)]
            cv2.putText(frame, f"NONE (transitions): weave {a.upper()} <-> {b.upper()}, keep moving",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            gate_ok = motion >= NONE_MIN_MOTION
            cv2.putText(frame,
                        f"samples: {counts[sign]}    hands: {n_hands}    "
                        f"motion: {motion:.4f} {'OK' if gate_ok else 'TOO STILL - not saving'}",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if gate_ok else (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"SIGN: {sign.upper()}  ({sign_idx+1}/{len(LABELS)})",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"samples: {counts[sign]}    hands: {n_hands}",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
        if hold_capture:
            cv2.putText(frame, "REC", (frame.shape[1] - 90, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif is_none_mode:
            # continuous capture is the whole point of none mode — shout when it's off
            cv2.putText(frame, "NOT RECORDING - press 'h' (focus this window)",
                        (frame.shape[1] - 620, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Naruto Sign Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            sign_idx = (sign_idx + 1) % len(LABELS)
            hold_capture = False
        elif key == ord("p"):
            sign_idx = (sign_idx - 1) % len(LABELS)
            hold_capture = False
        elif key == ord(" "):
            if n_hands > 0:
                append_sample(row, n_hands, sign, subject)
                counts[sign] += 1
        elif key == ord("h"):
            hold_capture = not hold_capture
        elif key == ord("d"):
            if delete_last_for_label(sign, subject):
                counts[sign] = max(0, counts[sign] - 1)
                print(f"Deleted last '{sign}' sample. Now {counts[sign]}.")
        elif key == ord("s"):
            print("\n--- sample counts ---")
            for s in LABELS:
                print(f"  {s:8s} {counts[s]}")
            print(f"  TOTAL    {sum(counts.values())}\n")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\nFinal counts:")
    for s in LABELS:
        print(f"  {s:8s} {counts[s]}")
    print(f"  TOTAL    {sum(counts.values())}")
    print(f"\nSaved to {CSV_PATH}")


if __name__ == "__main__":
    main()