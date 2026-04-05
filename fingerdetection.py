import cv2
import mediapipe as mp

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define finger tip landmarks (thumb, index, middle, ring, pinky)
tip_ids = [4, 8, 12, 16, 20]

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Process hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []

            # Get landmark positions
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            fingers = []

            # Thumb (compare x-coordinates)
            if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other 4 fingers (compare y-coordinates)
            for id in range(1, 5):
                if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

            # Display count on screen
            cv2.putText(frame, f'Fingers: {total_fingers}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show video feed
    cv2.imshow("Finger Counter", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()