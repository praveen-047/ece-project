import cv2

cap = cv2.VideoCapture(0)
try:
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
    else:
        print("Webcam is working. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nProgram interrupted. Releasing resources...")
finally:
    cap.release()
    cv2.destroyAllWindows()
