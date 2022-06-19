from itertools import chain
from copy import deepcopy
from typing import List

from pandas import DataFrame


def list_landmarks(landmarks) -> List[float]:
    landmark_points = []

    for landmark in landmarks.landmark:
        landmark_points.extend([landmark.x, landmark.y])

    return landmark_points


def add_landmarks_as_row(class_id: int, landmarks_list: List[float], df: DataFrame) -> None:
    list_to_insert = deepcopy(landmarks_list)
    list_to_insert.insert(0, class_id)
    df_length = len(df)
    df.loc[df_length] = list_to_insert
