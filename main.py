import cv2

from core.camera import Camera
from core.detector_manager import DetectorManager

from detectors.face_detector import FaceDetector

camera = Camera()

manager = DetectorManager()

manager.add_detector(FaceDetector())

while True:

    ret, frame = camera.read()

    if not ret:
        break

    frame = manager.process(frame)

    cv2.imshow("Vision", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
