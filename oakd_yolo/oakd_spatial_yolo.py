#!/usr/bin/env python

import contextlib
import datetime
import json
import math
import os
import time
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np

DISPLAY_WINDOW_SIZE_RATE = 2.0
MAX_Z = 15000
idColors = np.random.random(size=(256, 3)) * 256


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(
            frame, text, coords, self.text_type, 0.8, self.bg_color, 3, self.line_type
        )
        cv2.putText(
            frame, text, coords, self.text_type, 0.8, self.color, 1, self.line_type
        )

    def rectangle(self, frame, p1, p2, id):
        cv2.rectangle(frame, p1, p2, (0, 0, 0), 4)
        cv2.rectangle(frame, p1, p2, idColors[id], 2)


class HostSync:
    def __init__(self):
        self.dict = {}

    def add_msg(self, name, msg):
        seq = str(msg.getSequenceNum())
        if seq not in self.dict:
            self.dict[seq] = {}
        self.dict[seq][name] = msg

    def get_msgs(self):
        remove = []
        for name in self.dict:
            remove.append(name)
            if len(self.dict[name]) == 3:
                ret = self.dict[name]
                for rm in remove:
                    del self.dict[rm]
                return ret
        return None


class OakdSpatialYolo:
    def __init__(
        self,
        config_path: str,
        model_path: str,
        fps: int,
        fov: float,
        cam_debug: bool = False,
    ) -> None:
        if not Path(config_path).exists():
            raise ValueError("Path {} does not poetry exist!".format(config_path))
        with Path(config_path).open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

        # parse input shape
        if "input_size" in nnConfig:
            self.width, self.height = tuple(
                map(int, nnConfig.get("input_size").split("x"))
            )

        self.jet_custom = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET
        )
        self.jet_custom = self.jet_custom[::-1]
        self.jet_custom[0] = [0, 0, 0]
        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        self.classes = metadata.get("classes", {})
        self.coordinates = metadata.get("coordinates", {})
        self.anchors = metadata.get("anchors", {})
        self.anchorMasks = metadata.get("anchor_masks", {})
        self.iouThreshold = metadata.get("iou_threshold", {})
        self.confidenceThreshold = metadata.get("confidence_threshold", {})

        print(metadata)
        # parse labels
        nnMappings = config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})

        self.nn_path = Path(model_path)
        # get model path
        if not self.nn_path.exists():
            print(
                "No blob found at {}. Looking into DepthAI model zoo.".format(
                    self.nn_path
                )
            )
            self.nn_path = str(
                blobconverter.from_zoo(
                    model_path, shaves=6, zoo_type="depthai", use_cache=True
                )
            )
        self.fps = fps
        self.fov = fov
        self.cam_debug = cam_debug
        self._stack = contextlib.ExitStack()
        self._pipeline = self._create_pipeline()
        self._device = self._stack.enter_context(dai.Device(self._pipeline))
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        self.qRgb = self._device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.qDet = self._device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        self.qDepth = self._device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )
        self.counter = 0
        self.startTime = time.monotonic()
        self.frame_name = 0
        self.dir_name = ""
        self.path = ""
        self.is_save_start = False
        self.num = 0
        self.text = TextHelper()
        self.sync = HostSync()
        self.bird_eye_frame = self.create_bird_frame()

    def get_labels(self):
        return self.labels

    def _create_pipeline(self) -> dai.Pipeline:
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.initialControl.setManualFocus(130)
        camRgb.setPreviewKeepAspectRatio(False)

        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")

        # Properties
        camRgb.setPreviewSize(self.width, int(self.width * 9 / 16))
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(self.fps)

        # Use ImageMqnip to resize with letterboxing
        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(self.width * self.height * 3)
        manip.initialConfig.setResizeThumbnail(self.width, self.height)
        camRgb.preview.link(manip.inputImage)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        monoRight.setFps(self.fps)
        monoLeft.setFps(self.fps)

        spatialDetectionNetwork.setBlobPath(self.nn_path)
        spatialDetectionNetwork.setConfidenceThreshold(self.confidenceThreshold)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(300)
        spatialDetectionNetwork.setDepthUpperThreshold(35000)

        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(self.classes)
        spatialDetectionNetwork.setCoordinateSize(self.coordinates)
        spatialDetectionNetwork.setAnchors(self.anchors)
        spatialDetectionNetwork.setAnchorMasks(self.anchorMasks)
        spatialDetectionNetwork.setIouThreshold(self.iouThreshold)

        manip.out.link(spatialDetectionNetwork.input)
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutNN.setStreamName("nn")
        spatialDetectionNetwork.out.link(xoutNN.input)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
        return pipeline

    def frame_norm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def get_frame(self):
        frame = None
        detections = []
        if self.qRgb.has():
            self.sync.add_msg("rgb", self.qRgb.get())
        if self.qDepth.has():
            self.sync.add_msg("depth", self.qDepth.get())
        if self.qDet.has():
            self.sync.add_msg("detections", self.qDet.get())
            self.counter += 1
        msgs = self.sync.get_msgs()
        if msgs is not None:
            detections = msgs["detections"].detections
            frame = msgs["rgb"].getCvFrame()
            depthFrame = msgs["depth"].getFrame()
            depthFrameColor = cv2.normalize(
                depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3
            )
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, self.jet_custom)
            if self.cam_debug:
                cv2.imshow("rgb", cv2.resize(frame, (self.width, self.height)))
                cv2.imshow(
                    "depth",
                    cv2.resize(depthFrameColor, (self.width, int(self.width * 3 / 4))),
                )
        return frame, detections

    def display_frame(self, name, frame, detections):
        height = int(frame.shape[1] * 9 / 16)
        width = frame.shape[1]
        brank_height = width - height
        frame = frame[
            int(brank_height / 2) : int(frame.shape[0] - brank_height / 2),
            0:width,
        ]
        frame = cv2.resize(
            frame,
            (
                int(width * DISPLAY_WINDOW_SIZE_RATE),
                int(height * DISPLAY_WINDOW_SIZE_RATE),
            ),
        )
        display = frame
        # If the frame is available, draw bounding boxes on it and show the frame
        for detection in detections:
            # Fix ymin and ymax to cropped frame pos
            detection.ymin = (width / height) * detection.ymin - (
                brank_height / 2 / height
            )
            detection.ymax = (width / height) * detection.ymax - (
                brank_height / 2 / height
            )
            # Denormalize bounding box
            bbox = self.frame_norm(
                frame,
                (
                    detection.xmin,
                    detection.ymin,
                    detection.xmax,
                    detection.ymax,
                ),
            )
            x1 = bbox[0]
            x2 = bbox[2]
            y1 = bbox[1]
            y2 = bbox[3]
            try:
                label = self.labels[detection.label]
            except:
                label = detection.label
            self.text.putText(display, str(label), (x2 + 10, y1 + 20))
            self.text.putText(
                display,
                "{:.0f}%".format(detection.confidence * 100),
                (x2 + 10, y1 + 50),
            )
            self.text.rectangle(display, (x1, y1), (x2, y2), detection.label)
            if detection.spatialCoordinates.z != 0:
                self.text.putText(
                    display,
                    "X: {:.2f} m".format(detection.spatialCoordinates.x / 1000),
                    (x2 + 10, y1 + 80),
                )
                self.text.putText(
                    display,
                    "Y: {:.2f} m".format(detection.spatialCoordinates.y / 1000),
                    (x2 + 10, y1 + 110),
                )
                self.text.putText(
                    display,
                    "Z: {:.2f} m".format(detection.spatialCoordinates.z / 1000),
                    (x2 + 10, y1 + 140),
                )
            self.draw_bird_frame(
                detection.spatialCoordinates.x,
                detection.spatialCoordinates.z,
                detection.label
            )
        cv2.putText(
            display,
            "NN fps: {:.2f}".format(self.counter / (time.monotonic() - self.startTime)),
            (2, display.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.3,
            (255, 255, 255),
        )
        # Show the frame
        cv2.imshow(name, display)

    def create_bird_frame(self):
        fov = self.fov
        frame = np.zeros((300, 300, 3), np.uint8)
        cv2.rectangle(
            frame, (0, 283), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1
        )

        alpha = (180 - fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array(
            [
                (0, frame.shape[0]),
                (frame.shape[1], frame.shape[0]),
                (frame.shape[1], max_p),
                (center, frame.shape[0]),
                (0, max_p),
                (0, frame.shape[0]),
            ]
        )
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame

    def draw_bird_frame(self, x, z, id=None):
        birds = self.bird_eye_frame.copy()
        global MAX_Z
        max_x = MAX_Z / 2  # mm
        pointY = birds.shape[0] - int(z / (MAX_Z - 10000) * birds.shape[0]) - 20
        pointX = int(x / max_x * birds.shape[1] + birds.shape[1] / 2)
        if id is not None:
            # cv2.putText(frame, str(id), (pointX - 30, pointY + 5),
            #           cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
            cv2.circle(
                birds,
                (pointX, pointY),
                2,
                idColors[id],
                thickness=5,
                lineType=8,
                shift=0,
            )
        else:
            cv2.circle(
                birds,
                (pointX, pointY),
                2,
                (0, 255, 0),
                thickness=5,
                lineType=8,
                shift=0,
            )
        cv2.imshow("birds", birds)

    def save_image(self, frame):
        if not self.is_save_start:
            now = datetime.datetime.now()
            date = now.strftime("%Y%m%d%H%M")
            self.path = os.path.dirname(os.getcwd()) + "/data/" + date
            os.makedirs(self.path, exist_ok=True)
            self.is_save_start = True
        file_path = self.path + "/" + str(self.num).zfill(3) + ".jpg"
        cv2.imwrite(file_path, frame)
        print("save to: " + file_path)
        self.num += 1
