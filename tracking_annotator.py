#!/usr/bin/env python3
import argparse
import glob
import itertools
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from oakd_yolo.oakd_tracking_yolo import OakdTrackingYolo

# OAK-D LITEの視野角
fov = 56.7
save_num: int = 0
QUEUE_SIZE: int = 5


def main() -> None:
    global save_num
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Provide model name or model path for inference",
        default="yolov4_tiny_coco_416x416",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Provide config path for inference",
        default="json/yolov4-tiny.json",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fps",
        help="Camera frame fps. This should be smaller than nn inference fps",
        default=8,
        type=int,
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True, help="Path to save images"
    )
    args = parser.parse_args()
    os.makedirs(args.path, exist_ok=True)
    print(args.path)
    print(os.getcwd())
    path = os.getcwd() + "/" + args.path + "/"
    files = glob.glob(path + "/*")
    print(path)
    for file in files:
        file_name = Path(file).stem
        try:
            file_num = int(file_name)
            if file_num >= save_num:
                save_num = file_num + 1
        except:
            pass

    oakd_tracking_yolo = OakdTrackingYolo(args.config, args.model, args.fps, fov, False)
    track_pair: Tuple[Union[None, np.ndarray], Any] = (None, None)
    track_queue = deque()

    while True:
        frame = None
        detections = []
        frame, detections, tracklets = oakd_tracking_yolo.get_frame()
        if frame is not None:
            oakd_tracking_yolo.display_frame("nn", frame, tracklets)
            track_pair = (frame, tracklets)
            track_queue.append(track_pair)
            while len(track_queue) > QUEUE_SIZE:
                track_queue.popleft()
            name = str(save_num).zfill(3)
            if oakd_tracking_yolo.save_lost_frame(
                track_queue, path=path, name=name, save_tracked=True
            ):
                save_num += 1
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
