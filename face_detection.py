from detectors.base_detector import BaseDetector
import cv2

class FaceDetector(BaseDetector):

    def __init__(self):
        self.model = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.model.detectMultiScale(
            gray,
            1.1,
            4
        )

        return faces

    def draw(self, frame, faces):

        for x, y, w, h in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0,255,0),
                2
            )

        return frame
