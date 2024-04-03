#!/usr/bin/env python3
import argparse
import math
from typing import Any

import cv2
import numpy
import numpy as np
from lib.akari_yolo_lib.oakd_tracking_yolo import OakdTrackingYolo

# OAK-D LITEの視野角
fov = 56.7


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
    parser.add_argument(
        "-r",
        "--robot_coordinate",
        help="Convert object pos from camera coordinate to robot coordinate",
        action="store_true",
    )
    parser.add_argument(
        "--spatial_frame",
        help="Display spatial frame instead of bird frame",
        action="store_true",
    )
    parser.add_argument(
        "--orbit",
        help="Display tracked orbit on bird frame",
        action="store_true",
    )
    args = parser.parse_args()
    bird_frame = True
    spatial_frame = False
    # spatial_frameを有効化した場合、bird_frameは無効化
    if args.spatial_frame:
        bird_frame = False
        spatial_frame = True
    end = False

    while not end:
        oakd_tracking_yolo = OakdTrackingYolo(
            config_path=args.config,
            model_path=args.model,
            fps=args.fps,
            fov=fov,
            cam_debug=args.display_camera,
            robot_coordinate=args.robot_coordinate,
            show_bird_frame=bird_frame,
            show_spatial_frame=spatial_frame,
            show_orbit=args.orbit
        )
        while True:
            frame = None
            detections = []
            try:
                frame, detections, tracklets = oakd_tracking_yolo.get_frame()
            except BaseException:
                print("===================")
                print("get_frame() error! Reboot OAK-D.")
                print("If reboot occur frequently, Bandwidth may be too much.")
                print("Please lower FPS.")
                print("==================")
                break
            if frame is not None:
                oakd_tracking_yolo.display_frame("nn", frame, tracklets)
            if cv2.waitKey(1) == ord("q"):
                end = True
                break
        oakd_tracking_yolo.close()


if __name__ == "__main__":
    main()
