import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)
mp_draw = mp.solutions.drawing_utils

file_name = "hand_data.csv"
header = ["label"]
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

if not os.path.exists(file_name):
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("Collecting Data")
print("Instructions")
print("Hold corresponding letter to save data for that sign. Press 'q' to quit.")

target_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K']
counters = {letter: 0 for letter in target_letters}

data_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF
    key_char = chr(key).upper() if key != 255 else None

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_label = results.multi_handedness[i].classification[0].label
            if key_char in counters:
                landmarks = []
                
                for lm in hand_landmarks.landmark:
                    x_coor = 1.0 - lm.x if hand_label == "Left" else lm.x
                    landmarks.extend([x_coor, lm.y, lm.z])

                data_buffer.append([key_char] + landmarks)

                counters[key_char] += 1
                print(f"Captured {key_char} ({hand_label}): {counters[key_char]} samples", end="\r")

    if key_char in counters:
        count = counters[key_char]
        color = (0, 255, 0) if count >= 800 else (0, 255, 255)
        cv2.rectangle(frame, (10, 10), (280, 50), (0,0,0), -1)
        cv2.putText(frame, f"Recording '{key_char}': {count}/800", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(frame, "Ready - Hold a key", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("SignBridge Collector", frame)
    if key_char == 'Q':
        break

cap.release()
cv2.destroyAllWindows()

if data_buffer:
    print(f"Saving samples")
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_buffer)
print("finished")