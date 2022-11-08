import cv2
import mediapipe as mp
import sys
import math

class mediapipepose():
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic 

    def processImage(self,image):
        results = self.mp_holistic.Holistic(min_detection_confidence=0.1,min_tracking_confidence=0.1).process(image)
        
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, 
                                        self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, 
                                        self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image                               
