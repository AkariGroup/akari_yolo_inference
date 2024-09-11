#!/usr/bin/env python3
import argparse
import cv2
from lib.akari_yolo_lib.oakd_tracking_yolo import OakdTrackingYolo


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
    parser.add_argument(
        "--spatial_frame",
        help="Display spatial frame instead of bird frame",
        action="store_true",
    )
    parser.add_argument(
        "--disable_orbit",
        help="Disable display tracked orbit on bird frame",
        action="store_true",
    )
    parser.add_argument(
        "--log_path",
        help="Path to save orbit data",
        type=str,
    )
    parser.add_argument(
        "--log_continue",
        help="Continue log data",
        action="store_true",
    )
    args = parser.parse_args()
    bird_frame = True
    orbit = True
    spatial_frame = False
    if args.disable_orbit:
        orbit = False
    # spatial_frameを有効化した場合、bird_frameは無効化
    if args.spatial_frame:
        bird_frame = False
        spatial_frame = True
        orbit = False
    end = False

    while not end:
        oakd_tracking_yolo = OakdTrackingYolo(
            config_path=args.config,
            model_path=args.model,
            fps=args.fps,
            cam_debug=args.display_camera,
            robot_coordinate=args.robot_coordinate,
            show_bird_frame=bird_frame,
            show_spatial_frame=spatial_frame,
            show_orbit=orbit,
            log_path=args.log_path,
            log_continue=args.log_continue
        )
        oakd_tracking_yolo.update_bird_frame_distance(10000)
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
