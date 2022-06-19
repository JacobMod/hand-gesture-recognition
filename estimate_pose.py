import argparse

import cv2

from models import HandPoseEstimator


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source',
                    required=True,
                    type=int,
                    help='Camera source')

    return vars(ap.parse_args())


def main(args):
    vc = cv2.VideoCapture(args['source'])

    hands = HandPoseEstimator(2, 0.5)

    while True:
        ret, frame = vc.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        drawing_frame = frame.copy()
        results = hands(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hands.draw_keypoints(drawing_frame, hand_landmarks)

        cv2.imshow('Frame', drawing_frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser_args = make_parser()
    main(parser_args)
