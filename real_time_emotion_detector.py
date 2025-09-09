# real_time_emotion_detector.py
import cv2
from fer import FER

def initialize_camera():
    """Initialize webcam capture."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    return cap

def detect_emotions(frame, detector):
    """Detect emotions in a single frame."""
    result = detector.detect_emotions(frame)
    return result

def draw_emotions(frame, result):
    """Draw bounding boxes and emotion labels on the frame."""
    for face in result:
        (x, y, w, h) = face["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Find dominant emotion
        emotions = face["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        label = f"{dominant_emotion}: {confidence:.2f}"
        
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return frame

def main():
    """Main loop to capture webcam and show emotion detection."""
    print("Starting Real-Time Emotion Detector...")
    cap = initialize_camera()
    detector = FER(mtcnn=True)  # Use MTCNN for better face detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect emotions
        results = detect_emotions(frame, detector)
        
        # Draw results
        output_frame = draw_emotions(frame, results)
        
        # Display
        cv2.imshow("Real-Time Emotion Detector", output_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Emotion Detector stopped.")

if __name__ == "__main__":
    main()
