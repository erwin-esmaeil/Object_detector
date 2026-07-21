import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open camera (0 is default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection on the current frame
    results = model(frame, stream=True)

    # Visualize results on the frame
    for r in results:
        annotated_frame = r.plot()
        
    # Display the frame
    cv2.imshow("YOLO Live Detection", annotated_frame)

    # Press 'q' on the keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()





