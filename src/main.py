# src/main.py

import cv2
from emotion_detector import detect_faces, predict_emotion
from joke_generator import cheer_user

cap = cv2.VideoCapture(0)

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emotion = predict_emotion(face)

        # Draw rectangle + emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, emotion, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        # Trigger joke only if emotion changed
        if emotion != last_emotion:
            cheer_user(emotion)
            last_emotion = emotion

    cv2.imshow("EmotionCheer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
