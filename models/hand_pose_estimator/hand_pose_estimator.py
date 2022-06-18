import mediapipe as mp


class HandPoseEstimator:
    def __init__(self,
                 hands_number: int,
                 detection_confidence: float):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True,
                                         max_num_hands=hands_number,
                                         min_detection_confidence=detection_confidence)

    def __call__(self, image):
        return self.hands.process(image)

    def draw_keypoints(self, image, hand_landmarks):
        self.mp_drawing.draw_landmarks(image,
                                       hand_landmarks,
                                       self.mp_hands.HAND_CONNECTIONS,
                                       self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                       self.mp_drawing_styles.get_default_hand_connections_style())
