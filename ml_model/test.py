import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

labels = {i: char for i, char in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K'])}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_label = results.multi_handedness[i].classification[0].label
            landmarks = []
            for lm in hand_landmarks.landmark:
                x_coor = 1.0 - lm.x if hand_label == "Left" else lm.x
                landmarks.extend([x_coor, lm.y, lm.z])
            
            input_data = np.array([landmarks], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            prediction = np.argmax(output_data)
            confidence = output_data[0][prediction]

            if confidence > 0.8:
                cv2.putText(frame, f"Sign: {labels[prediction]} ({int(confidence*100)}%) | Hand: {hand_label}", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("SignBridge Live Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()