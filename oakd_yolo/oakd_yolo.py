#!/usr/bin/env python

import contextlib
import datetime
import json
import os
import time
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np

WIDTH = 640
HEIGHT = 640
BLACK = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
DISPLAY_WINDOW_SIZE = 1.5


class OakdYolo:
    def __init__(self, configPath: str, modelPath: str) -> None:
        if not Path(configPath).exists():
            raise ValueError("Path {} does not poetry exist!".format(configPath))
        with Path(configPath).open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

        # parse input shape
        if "input_size" in nnConfig:
            W, H = tuple(map(int, nnConfig.get("input_size").split("x")))

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

        self.nnPath = Path(modelPath)
        # get model path
        if not self.nnPath.exists():
            print(
                "No blob found at {}. Looking into DepthAI model zoo.".format(
                    self.nnPath
                )
            )
            self.nnPath = str(
                blobconverter.from_zoo(
                    modelPath, shaves=6, zoo_type="depthai", use_cache=True
                )
            )

        self._stack = contextlib.ExitStack()
        self._pipeline = self._create_pipeline()
        self._device = self._stack.enter_context(
            dai.Device(self._pipeline, usb2Mode=True)
        )
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        self.qControl = self._device.getInputQueue("control")
        self.qRgb = self._device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.qIsp = self._device.getOutputQueue(name="isp")
        self.qDet = self._device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        self.counter = 0
        self.startTime = time.monotonic()
        self.frame_name = 0
        self.dir_name = ""
        self.path = ""
        self.is_save_start = False
        self.num = 0
        self.startTime = time.monotonic()
        self.counter = 0

    def set_camera_brightness(self, brightness: int) -> None:
        ctrl = dai.CameraControl()
        ctrl.setBrightness(brightness)
        self.qControl.send(ctrl)

    def get_labels(self):
        return self.labels

    def _create_pipeline(self) -> dai.Pipeline:
        # OAK-Dのセットアップ
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        controlIn = pipeline.create(dai.node.XLinkIn)
        camRgb = pipeline.create(dai.node.ColorCamera)
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        nnOut = pipeline.create(dai.node.XLinkOut)
        controlIn.setStreamName("control")
        xoutRgb.setStreamName("rgb")
        nnOut.setStreamName("nn")

        # Properties
        controlIn.out.link(camRgb.inputControl)
        camRgb.setPreviewKeepAspectRatio(False)
        # camRgb.setPreviewSize(W, H)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        # camRgb.setIspScale(1, 2)
        camRgb.setPreviewSize(1920, 1080)
        camRgb.setFps(10)

        xoutIsp = pipeline.create(dai.node.XLinkOut)
        xoutIsp.setStreamName("isp")
        camRgb.isp.link(xoutIsp.input)

        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(WIDTH * HEIGHT * 3)  # 640x640x3
        manip.initialConfig.setResizeThumbnail(WIDTH, HEIGHT)
        camRgb.preview.link(manip.inputImage)

        # Network specific settings
        detectionNetwork.setConfidenceThreshold(self.confidenceThreshold)
        detectionNetwork.setNumClasses(self.classes)
        detectionNetwork.setCoordinateSize(self.coordinates)
        detectionNetwork.setAnchors(self.anchors)
        detectionNetwork.setAnchorMasks(self.anchorMasks)
        detectionNetwork.setIouThreshold(self.iouThreshold)
        detectionNetwork.setBlobPath(self.nnPath)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)

        # Linking
        # camRgb.preview.link(detectionNetwork.input)
        manip.out.link(detectionNetwork.input)
        detectionNetwork.passthrough.link(xoutRgb.input)
        detectionNetwork.out.link(nnOut.input)
        return pipeline

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def getFrame(self):
        inRgb = self.qRgb.get()
        inIsp = self.qIsp.get()
        inDet = self.qDet.get()

        if inIsp is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(
                    self.counter / (time.monotonic() - self.startTime)
                ),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                BLACK,
            )

        if inDet is not None:
            detections = inDet.detections
            self.counter += 1
        return frame, detections

    def displayFrame(self, name, frame, detections, focus_index):
        width = frame.shape[1]
        height = frame.shape[1] * 9 / 16
        brank_height = width - height
        frame = frame[
            int(brank_height / 2) : int(frame.shape[0] - brank_height / 2), 0:width
        ]
        frame = cv2.resize(
            frame, (int(width * DISPLAY_WINDOW_SIZE), int(height * DISPLAY_WINDOW_SIZE))
        )
        for i in range(0, len(detections)):
            detections[i].ymin = (width / height) * detections[i].ymin - (
                brank_height / 2 / height
            )
            detections[i].ymax = (width / height) * detections[i].ymax - (
                brank_height / 2 / height
            )
            if i == focus_index:
                color = GREEN
            else:
                color = BLUE
            bbox = self.frameNorm(
                frame,
                (
                    detections[i].xmin,
                    detections[i].ymin,
                    detections[i].xmax,
                    detections[i].ymax,
                ),
            )
            cv2.putText(
                frame,
                self.labels[detections[i].label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )
            cv2.putText(
                frame,
                f"{int(detections[i].confidence * 100)}%",
                (bbox[0] + 10, bbox[1] + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(
            frame,
            "NN fps: {:.2f}".format(self.counter / (time.monotonic() - self.startTime)),
            (2, frame.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            (255, 255, 255),
        )
        # Show the frame
        cv2.imshow(name, frame)

    def saveImage(self, frame):
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
