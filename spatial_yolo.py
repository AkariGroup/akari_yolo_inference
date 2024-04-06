#!/usr/bin/env python3
import argparse
import cv2
from lib.akari_yolo_lib.oakd_spatial_yolo import OakdSpatialYolo


def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Provide model name or model path for inference",
        default="yolov7tiny_coco_416x416",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Provide config path for inference",
        default="json/yolov7tiny_coco_416x416.json",
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
    args = parser.parse_args()

    end = False
    while not end:
        oakd_spatial_yolo = OakdSpatialYolo(
            config_path=args.config,
            model_path=args.model,
            fps=args.fps,
            cam_debug=args.display_camera,
            robot_coordinate=args.robot_coordinate,
        )
        while True:
            frame = None
            detections = []
            try:
                frame, detections = oakd_spatial_yolo.get_frame()
            except BaseException:
                print("===================")
                print("get_frame() error! Reboot OAK-D.")
                print("If reboot occur frequently, Bandwidth may be too much.")
                print("Please lower FPS.")
                print("==================")
                break
            if frame is not None:
                oakd_spatial_yolo.display_frame("nn", frame, detections)
            if cv2.waitKey(1) == ord("q"):
                end = True
                break
        oakd_spatial_yolo.close()


if __name__ == "__main__":
    main()
