import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def finger_is_up(lm, tip_id, pip_id):
    return lm[tip_id][1] < lm[pip_id][1]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from camera")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = np.array([[pt.x, pt.y] for pt in hand_landmarks.landmark])

            index_up = finger_is_up(lm, 8, 6)
            middle_up = finger_is_up(lm, 12, 10)
            ring_up = finger_is_up(lm, 16, 14)
            pinky_up = finger_is_up(lm, 20, 18)
            thumb_up = lm[4][0] > lm[3][0]

            fingers = [thumb_up, index_up, middle_up, ring_up, pinky_up]
            count = sum(fingers)

            if count == 0:
                gesture = "✊ Fist"
            elif count == 5:
                gesture = "🖐️ Open Palm"
            elif count == 2 and index_up and middle_up and not ring_up and not pinky_up:
                gesture = "✌️ Peace"
            elif count == 1 and thumb_up and not any([index_up, middle_up, ring_up, pinky_up]):
                gesture = "👍 Thumbs Up"
            else:
                gesture = f"🤷 Unknown ({count} finger(s) up)"

    cv2.putText(frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Hand Gesture Recognition", frame)
    # Debug window (optional)
    # cv2.imshow("Raw Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()