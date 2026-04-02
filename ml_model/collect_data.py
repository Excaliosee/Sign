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
print("Hold 0-4 to save data for that sign. Press 'q' to quit.")

counters = {str(i): 0 for i in range(5)}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            key = cv2.waitKey(1) & 0xFF
            if chr(key) in counters:
                label = chr(key)
                with open(file_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + landmarks)

                counters[label] += 1
                print(f"Caputed {label}: {counters[label]} samples", end="\r")

    y_pos = 30
    for label, count in counters.items():
        cv2.putText(frame, f"Sign {label}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y_pos += 25

    cv2.imshow("SignBridge Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("finished")