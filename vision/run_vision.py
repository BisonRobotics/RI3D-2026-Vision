import ntcore
import cv2
from pupil_apriltags import Detector
import logging
import sys
import time
from cscore import CameraServer
import numpy as np
import json


# TODO: Graceful shutdown with sigterm
# TODO: Clean up classes
# TODO: Stub image stream on environment var
# TODO: Capture class


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Starting Vision. . .")


class TagDetector:
    def __init__(self, families="tag36h11"):
        self.logger = logging.getLogger()
        self.families = families
        self.detector = self._make_detector(families)
        self.results = []

    def _make_detector(self, families: str) -> Detector:
        # Tune as needed; these defaults are sane and stable on Pi.
        return Detector(
            families=families,
            quad_decimate=1.0,     # >1.0 speeds up, reduces accuracy
            quad_sigma=0.0,        # blur; usually keep 0
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def set_families(self, families: str):
        self.families = families
        self.detector = self._make_detector(families)

    def detect(self, frame):
        """
        Expects a grayscale uint8 image (H x W), contiguous.
        Returns the list of detections.
        """
        self.results = self.detector.detect(frame)
        return self.results

    def draw_boxes(self, frame):
        """
        Draws tag outlines + centers + family label.
        Expects BGR image in OpenCV format.
        """
        for r in self.results:
            # corners is (4,2): [ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]
            corners = np.array(r.corners, dtype=np.float32)
            pts = corners.astype(int)

            # Draw outline
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
            cv2.line(frame, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), 2)
            cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (0, 255, 0), 2)
            cv2.line(frame, tuple(pts[3]), tuple(pts[0]), (0, 255, 0), 2)

            # Center
            cX, cY = int(r.center[0]), int(r.center[1])
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

            # Family label near first corner
            tag_family = r.tag_family
            if isinstance(tag_family, bytes):
                tag_family = tag_family.decode("utf-8", errors="ignore")
            elif not isinstance(tag_family, str):
                tag_family = str(tag_family)

            # Add ID too (usually helpful)
            label = f"{tag_family}:{r.tag_id}"
            x0, y0 = int(pts[0][0]), int(pts[0][1])
            cv2.putText(
                frame,
                label,
                (x0, max(0, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame

    def get_centers(self):
        return [(float(r.center[0]), float(r.center[1])) for r in self.results]

    def get_ids(self):
        return [int(r.tag_id) for r in self.results]
    
class PublisherServer:
    def __init__(self):
        self.inst = ntcore.NetworkTableInstance.getDefault()
        
    def start(self):
        self.inst.startServer(listen_address="0.0.0.0", port4=ntcore.NetworkTableInstance.kDefaultPort4)

    def get_table(self, name):
        return self.inst.getTable(name)


class ImagePublisher:
    def __init__(self, table):
        self.table = table
        self.pub = table.getRawTopic("image").publish("image/jpeg")

    def set(self, frame: bytes):
        self.pub.set(frame)

class IdPublisher:
    def __init__(self, table):
        self.table = table
        self.pub = table.getIntegerArrayTopic("ids").publish()

    def set(self, ids):
        self.pub.set(ids)

class JsonStringPublisher:
    def __init__(self, table):
        self.table = table
        self.pub = table.getStringTopic("json").publish()

    def set(self, data):
        json_str = json.dumps(data)
        self.pub.set(json_str)


def main():
    setup_logger() # Setup up global logger
    logger  = logging.getLogger() # Get instance of logger
    detector = TagDetector()
    server = PublisherServer()
    server.start()
    vision_table = server.get_table("Vision")
    img_pub = ImagePublisher(vision_table)
    string_pub = JsonStringPublisher(vision_table)

    # capture = cv2.VideoCapture(2)
    img = np.zeros(shape=(720, 1280, 3), dtype=np.uint8)
    camera = CameraServer.startAutomaticCapture(0)
    camera.setResolution(1280, 720)
    camera.setFPS(30)
    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', 1280, 720)

    # April tag data is stored here and then published as JSON.
    # Data shall be stored as:
    # {
    #     "id": {
    #         "center": [x, y],
    #         "corners": [ [x0, y0], [x1, y1], [x2, y2], [x3, y3] ],
    #     },
    #     ...
    # }
    tag_data = {}

    while True:
        # ret, frame = capture.read()
        frame_time, frame = input_stream.grabFrame(img)
        output_img = np.copy(frame)
        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)  # <-- critical
        results = detector.detect(gray)
        frame = detector.draw_boxes(output_img)
        for result in results:
            tag_id = int(result.tag_id)
            center = [float(result.center[0]), float(result.center[1])]
            corners = [[float(c[0]), float(c[1])] for c in result.corners]
            tag_data[tag_id] = {
                "center": center,
                "corners": corners,
            }
            string_pub.set(tag_data)

        # Write image to UI
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ok:
            img_pub.set(jpg.tobytes())

        # Write image to output stream
        output_stream.putFrame(output_img)

        time.sleep(1/30)  # Simulate 30 FPS


if __name__ == "__main__":
    main()
