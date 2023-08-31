#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import math
from typing import Any
from collections import deque
import itertools

import cv2
import numpy
import numpy as np
from pathlib import Path

from oakd_yolo.oakd_tracking_yolo import OakdTrackingYolo

# OAK-D LITEの視野角
fov = 56.7
save_num: int = 0
QUEUE_SIZE: int = 5


def tracklets_to_annotation(tracklets: Any):
    annotation_text = ""
    for tracklet in tracklets:
        center_x: float = (tracklet.roi.topLeft().x + tracklet.roi.bottomRight().x) / 2
        center_y: float = (tracklet.roi.topLeft().y + tracklet.roi.bottomRight().y) / 2
        width: float = tracklet.roi.bottomRight().x - tracklet.roi.topLeft().x
        height: float = tracklet.roi.bottomRight().y - tracklet.roi.topLeft().y
        annotation_text += f"{tracklet.label} {center_x} {center_y} {width} {height}\n"
    return annotation_text


def save_lost_frame(queue, path: str, save_tracked: bool = True) -> None:
    global save_num
    if len(queue) <= 2:
        return
    save_frame: bool = False
    save_track = []
    for tracklet in queue[-2][1]:
        if tracklet.status.name == "TRACKED":
            if save_tracked:
                save_track.append(tracklet)
        elif tracklet.status.name == "LOST":
            # LOSTの場合は最新のフレームがTRACKEDに復帰しているか見る
            for now_tracklet in queue[-1][1]:
                if (
                    now_tracklet.id == tracklet.id
                    and now_tracklet.status.name == "TRACKED"
                ):
                    # フレーム頭から2個前のフレームまでの間にTRACKEDの状態であれば、保存する
                    prev_pairs = itertools.islice(queue, 0, len(queue) - 2)
                    tracklet_saved = False
                    for prev in prev_pairs:
                        if(tracklet_saved):
                            break
                        for prev_tracklet in prev[1]:
                            if (
                                prev_tracklet.id == tracklet.id
                                and prev_tracklet.status.name == "TRACKED"
                            ):
                                save_frame = True
                                tracklet_saved = True
                                save_track.append(tracklet)
                                break
    if save_frame:
        save_path: str = path + "/" + str(save_num).zfill(3)
        image_path = save_path + ".jpg"
        cv2.imwrite(image_path, queue[-1][0])
        annotation: str = tracklets_to_annotation(save_track)
        annotation_path = save_path + ".txt"
        with open(annotation_path, "w") as file:
            file.write(annotation)
        print(f"save {str(save_num).zfill(3)}")
        save_num += 1


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
    track_pair = None
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
            save_lost_frame(track_queue, path=path, save_tracked=True)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
