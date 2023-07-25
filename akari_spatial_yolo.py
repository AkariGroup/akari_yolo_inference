#!/usr/bin/env python3
import argparse
import json
import math
import time
import numpy
from pathlib import Path
from typing import Any

import blobconverter
import cv2
import depthai as dai
import numpy as np
from oakd_yolo.oakd_spatial_yolo import OakdSpatialYolo
from akari_client import AkariClient

MAX_Z = 15000
DISPLAY_WINDOW_SIZE_RATE = 3.0
idColors = np.random.random(size=(256, 3)) * 256
fov = 56.7
akari = AkariClient()
joints = akari.joints


def convert_to_pos_from_akari(pos: Any, pitch: float, yaw: float) -> Any:
    cur_pos = np.array([[pos.x], [pos.y], [pos.z]])
    arr_y = np.array(
        [
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)]
        ]
    )
    arr_p = np.array(
        [
            [1, 0, 0],
            [
                0,
                math.cos(pitch),
                -math.sin(pitch),
            ],
            [0, math.sin(pitch), math.cos(pitch)]
        ]
    )
    ans = arr_y @ arr_p @ cur_pos
    return ans


def main() -> None:

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
        default=10,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--display_camera",
        help="Display camera rgb and depth frame",
        action="store_true",
    )
    args = parser.parse_args()

    # parse config
    configPath = Path(args.config)
    if not configPath.exists():
        raise ValueError("Path {} does not exist!".format(configPath))

    # get model path
    nnPath = args.model
    if not Path(nnPath).exists():
        print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
        nnPath = str(
            blobconverter.from_zoo(
                args.model, shaves=6, zoo_type="depthai", use_cache=True
            )
        )

    oakd_spatial_yolo = OakdSpatialYolo(
        configPath, nnPath, args.fps, fov, args.display_camera
    )
    frame = None
    detections = []

    while True:
        frame, detections = oakd_spatial_yolo.get_frame()
        head_pos = joints.get_joint_positions()
        pitch = head_pos["tilt"]
        yaw = head_pos["pan"]
        if frame is not None:
            for detection in detections:
                converted_pos = convert_to_pos_from_akari(
                    detection.spatialCoordinates, -1 * pitch, -1 * yaw
                )
                detection.spatialCoordinates.x = converted_pos[0]
                detection.spatialCoordinates.y = converted_pos[1]
                detection.spatialCoordinates.z = converted_pos[2]
                oakd_spatial_yolo.display_frame("nn", frame, detections)
        if cv2.waitKey(1) == ord("q"):
            break




if __name__ == "__main__":
    main()
