import argparse
from enum import IntEnum

import cv2
import pandas as pd

from models import HandPoseEstimator

from utils import list_files_w_extensions
from utils.landmarks import list_landmarks, add_landmarks_as_row


class Signs(IntEnum):
    SHAKA = 0
    OK = 1
    OPEN_HAND = 2
    CLOSED_HAND = 3


COLUMNS_NAMES = ["class_id",
                 "wrist.x",
                 "wrist.y",
                 "thumb_cmc.x",
                 "thumb_cmc.y",
                 "thumb_mcp.x",
                 "thumb_mcp.y",
                 "thumb_ip.x",
                 "thumb_ip.y",
                 "thumb_tip.x",
                 "thumb_tip.y",
                 "index_finger_mcp.x",
                 "index_finger_mcp.y",
                 "index_finder_pip.x",
                 "index_finder_pip.y",
                 "index_finger_dip.x",
                 "index_finger_dip.y",
                 "index_finder_tip.x",
                 "index_finder_tip.y",
                 "middle_finger_mcp.x",
                 "middle_finger_mcp.y",
                 "middle_finger_pip.x",
                 "middle_finger_pip.y",
                 "middle_finger_dip.x",
                 "middle_finger_dip.y",
                 "middle_finger_tip.x",
                 "middle_finger_tip.y",
                 "ring_finger_mcp.x",
                 "ring_finger_mcp.y",
                 "ring_finger_pip.x",
                 "ring_finger_pip.y",
                 "ring_finger_dip.x",
                 "ring_finger_dip.y",
                 "ring_finger_tip.x",
                 "ring_finger_tip.y",
                 "pinky_mcp.x",
                 "pinky_mcp.y",
                 "pinky_pip.x",
                 "pinky_pip.y",
                 "pinky_dip.x",
                 "pinky_dip.y",
                 "pinky_tip.x",
                 "pinky_tip.y"]


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images',
                    required=True,
                    type=str,
                    help='Images folder path')
    ap.add_argument('--data-file',
                    required=False,
                    type=str,
                    help='Annotation file, if not provided then it will be created and'
                         'saved in dataset/gesture. file')
    return vars(ap.parse_args())


def main(args):
    images_paths = list_files_w_extensions(args['images'], ['.jpg',
                                                            '.png',
                                                            '.JPG',
                                                            '.JPEG'])
    pose_estimator = HandPoseEstimator(1, 0.5)
    df = pd.read_csv(args['data_file']) if args['data_file'] else pd.DataFrame(columns=COLUMNS_NAMES)
    annotation_filename = args['data_file'] if args['data_file'] else './dataset/gestures.csv'
    keys = [s.value for s in Signs]

    for image_path in images_paths:
        image = cv2.imread(image_path)
        image = cv2.flip(image, 1)
        drawing_image = image.copy()
        results = pose_estimator(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks is None:
            continue

        for hand_landmarks in results.multi_hand_landmarks:
            pose_estimator.draw_keypoints(drawing_image, hand_landmarks)
            cv2.imshow('Frame', drawing_image)
            k = cv2.waitKey(0)
            landmarks_list = list_landmarks(hand_landmarks)

            if k == ord('0'):
                add_landmarks_as_row(Signs.SHAKA.value, landmarks_list, df)
                print(f"Annotated class {Signs.SHAKA.name} for image {image_path}")
            elif k == ord('1'):
                add_landmarks_as_row(Signs.OK.value, landmarks_list, df)
                print(f"Annotated class {Signs.OK.name} for image {image_path}")
            elif k == ord('2'):
                add_landmarks_as_row(Signs.OPEN_HAND.value, landmarks_list, df)
                print(f"Annotated class {Signs.OPEN_HAND.name} for image {image_path}")
            elif k == ord('3'):
                add_landmarks_as_row(Signs.CLOSED_HAND.value, landmarks_list, df)
                print(f"Annotated class {Signs.CLOSED_HAND.name} for image {image_path}")
            else:
                continue

        df.to_csv(annotation_filename, index=False)
        print(f"Saved file {annotation_filename}")


if __name__ == '__main__':
    parser_args = make_parser()
    main(parser_args)
