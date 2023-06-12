#!/usr/bin/env python3
from pathlib import Path
import sys
import math
import cv2
import depthai as dai
import numpy as np
import argparse
import json
import blobconverter
import time


MAX_Z = 15000
DISPLAY_WINDOW_SIZE_RATE = 3.0
idColors = np.random.random(size=(256, 3)) * 256


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type,
                    1, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type,
                    1, self.color, 1, self.line_type)

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


def create_bird_frame():
    fov = 72.9
    frame = np.zeros((300, 100, 3), np.uint8)
    cv2.rectangle(frame, (0, 283),
                  (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

    alpha = (180 - fov) / 2
    center = int(frame.shape[1] / 2)
    max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
    fov_cnt = np.array([
        (0, frame.shape[0]),
        (frame.shape[1], frame.shape[0]),
        (frame.shape[1], max_p),
        (center, frame.shape[0]),
        (0, max_p),
        (0, frame.shape[0]),
    ])
    cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
    return frame


def draw_bird_frame(frame, x, z, id=None):
    global MAX_Z
    max_x = 5000  # mm
    pointY = frame.shape[0] - int(z / (MAX_Z - 10000) * frame.shape[0]) - 20
    pointX = int(-x / max_x * frame.shape[1] + frame.shape[1] / 2)
    if id is not None:
        # cv2.putText(frame, str(id), (pointX - 30, pointY + 5),
        #           cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
        cv2.circle(frame, (pointX, pointY), 2, idColors[id],
                   thickness=5, lineType=8, shift=0)
    else:
        cv2.circle(frame, (pointX, pointY), 2, (0, 255, 0),
                   thickness=5, lineType=8, shift=0)


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def main() -> None:
    birdseyeframe = create_bird_frame()
    jet_custom = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    jet_custom = jet_custom[::-1]
    jet_custom[0] = [0, 0, 0]

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                        default='yolov4_tiny_coco_416x416', type=str)
    parser.add_argument("-c", "--config", help="Provide config path for inference",
                        default='json/yolov4-tiny.json', type=str)
    parser.add_argument("-f", "--fps", help="Camera frame fps. This should be smaller than nn inference fps",
                        default=10, type=int)
    args = parser.parse_args()

    # parse config
    configPath = Path(args.config)
    if not configPath.exists():
        raise ValueError("Path {} does not exist!".format(configPath))

    with configPath.open() as f:
        config = json.load(f)
    nnConfig = config.get("nn_config", {})

    # parse input shape
    if "input_size" in nnConfig:
        W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

    # extract metadata
    metadata = nnConfig.get("NN_specific_metadata", {})
    classes = metadata.get("classes", {})
    coordinates = metadata.get("coordinates", {})
    anchors = metadata.get("anchors", {})
    anchorMasks = metadata.get("anchor_masks", {})
    iouThreshold = metadata.get("iou_threshold", {})
    confidenceThreshold = metadata.get("confidence_threshold", {})

    nnMappings = config.get("mappings", {})
    labels = nnMappings.get("labels", {})

    # get model path
    nnPath = args.model
    if not Path(nnPath).exists():
        print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
        nnPath = str(blobconverter.from_zoo(
            args.model, shaves=6, zoo_type="depthai", use_cache=True))

    syncNN = False
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.initialControl.setManualFocus(130)
    # camRgb.setIspScale(2, 3)  # Downscale color to match mono
    camRgb.setPreviewKeepAspectRatio(False)

    spatialDetectionNetwork = pipeline.create(
        dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    # Properties
    camRgb.setPreviewSize(1920, 1080)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(args.fps)

    xoutIsp = pipeline.create(dai.node.XLinkOut)
    xoutIsp.setStreamName("isp")
    camRgb.isp.link(xoutIsp.input)

    # Use ImageMqnip to resize with letterboxing
    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(W * H * 3)
    manip.initialConfig.setResizeThumbnail(W, H)
    camRgb.preview.link(manip.inputImage)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # setting node configs
    stereo.setDefaultProfilePreset(
        dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    monoRight.setFps(args.fps)
    monoLeft.setFps(args.fps)

    spatialDetectionNetwork.setBlobPath(nnPath)
    spatialDetectionNetwork.setConfidenceThreshold(confidenceThreshold)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(300)
    spatialDetectionNetwork.setDepthUpperThreshold(35000)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(classes)
    spatialDetectionNetwork.setCoordinateSize(coordinates)
    spatialDetectionNetwork.setAnchors(anchors)
    spatialDetectionNetwork.setAnchorMasks(anchorMasks)
    spatialDetectionNetwork.setIouThreshold(iouThreshold)

    # camRgb.preview.link(spatialDetectionNetwork.input)
    manip.out.link(spatialDetectionNetwork.input)
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    stereo.depth.link(spatialDetectionNetwork.inputDepth)

    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutNN.setStreamName("detections")
    spatialDetectionNetwork.out.link(xoutNN.input)

    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb")
        detectionNNQueue = device.getOutputQueue(name="detections")
        depthQueue = device.getOutputQueue(name="depth")
        manipQueue = device.getOutputQueue(name='isp')

        text = TextHelper()
        sync = HostSync()
        display = None
        startTime = time.monotonic()
        counter = 0

        while True:
            if previewQueue.has():
                sync.add_msg("rgb", previewQueue.get())
            if depthQueue.has():
                sync.add_msg("depth", depthQueue.get())
            if manipQueue.has():
                manipQueue.get()
            if detectionNNQueue.has():
                sync.add_msg("detections", detectionNNQueue.get())
                counter += 1

            msgs = sync.get_msgs()
            if msgs is not None:
                birds = birdseyeframe.copy()
                detections = msgs["detections"].detections
                frame = msgs["rgb"].getCvFrame()
                depthFrame = msgs["depth"].getFrame()
                depthFrameColor = cv2.normalize(
                    depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(
                    depthFrameColor, jet_custom)
                # Crop the frame Square to 16:9
                height = frame.shape[1] * 9 / 16
                width = frame.shape[1]
                brank_height = width - height
                frame = frame[int(brank_height / 2): int(frame.shape[0] -
                                                         brank_height / 2), 0:width]
                frame = cv2.resize(frame, (int(
                    width * DISPLAY_WINDOW_SIZE_RATE), int(height * DISPLAY_WINDOW_SIZE_RATE)))
                display = frame
                display = cv2.flip(display, 1)
                # If the frame is available, draw bounding boxes on it and show the frame
                for detection in detections:
                    # Fix ymin and ymax to cropped frame pos
                    detection.ymin = ((width / height) *
                                      detection.ymin - (brank_height / 2 / height))
                    detection.ymax = ((width / height) *
                                      detection.ymax - (brank_height / 2 / height))
                    # Denormalize bounding box
                    detection.xmin = 1 - detection.xmin
                    detection.xmax = 1 - detection.xmax
                    bbox = frameNorm(
                        frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    x1 = bbox[0]
                    x2 = bbox[2]
                    y1 = bbox[1]
                    y2 = bbox[3]
                    try:
                        label = labels[detection.label]
                    except:
                        label = detection.label
                    text.putText(display, str(label), (x2 + 10, y1 + 20))
                    text.putText(display, "{:.0f}%".format(
                        detection.confidence * 100), (x2 + 10, y1 + 50))
                    text.rectangle(display, (x1, y1),
                                   (x2, y2), detection.label)
                    if detection.spatialCoordinates.z != 0:
                        text.putText(display, "X: {:.2f} m".format(
                            detection.spatialCoordinates.x / 1000), (x2 + 10, y1 + 80))
                        text.putText(display, "Y: {:.2f} m".format(
                            detection.spatialCoordinates.y / 1000), (x2 + 10, y1 + 110))
                        text.putText(display, "Z: {:.2f} m".format(
                            detection.spatialCoordinates.z / 1000), (x2 + 10, y1 + 140))
                    draw_bird_frame(birds, detection.spatialCoordinates.x,
                                    detection.spatialCoordinates.z, detection.label)
                cv2.putText(display, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                            (2, display.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

            if display is not None:
                # Birdseye view
                cv2.imshow("birds", birds)
                cv2.imshow("yolo", cv2.resize(display, (960, 540)))

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
