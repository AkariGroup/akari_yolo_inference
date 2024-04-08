#!/usr/bin/env python3
import argparse
import cv2
from lib.akari_yolo_lib.oakd_tracking_yolo import OrbitPlayer


def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--speed",
        help="Play speed",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-l",
        "--log_path",
        help="Log path to play",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--max_z",
        help="Max z distance in frame [mm]",
        default=15000,
        type=float,
    )
    args = parser.parse_args()
    orbit_player = OrbitPlayer(
        log_path=args.log_path, speed=args.speed, max_z=args.max_z
    )
    orbit_player.play_log()


if __name__ == "__main__":
    main()
