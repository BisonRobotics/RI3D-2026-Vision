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


# TODO: Add more fields as needed.
# Additional output fields (to include with tag_data / published JSON):
# rawBytes (byte[]): A byte-packed string that contains target info from the same timestamp.
# latencyMillis (double): The latency of the pipeline in milliseconds.
# hasTarget (boolean): Whether the pipeline is detecting targets or not.
# targetPitch (double): The pitch of the target in degrees (positive up).
# targetYaw (double): The yaw of the target in degrees (positive right).
# targetArea (double): The area (percent of bounding box in screen) as a percent (0-100).
# targetSkew (double): The skew of the target in degrees (counter-clockwise positive).
# targetPose (double[]): The pose of the target relative to the robot (x, y, z, qw, qx, qy, qz).
# targetPixelsX (double): The target crosshair location horizontally, in pixels (origin top-right).
# targetPixelsY (double): The target crosshair location vertically, in pixels (origin top-right).
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
    
    def get_corners(self):
        return [r.corners for r in self.results]
    
    def get_pitch_yaw(self):
        pitch_yaw_list = []
        for r in self.results:
            # Calculate pitch and yaw based on the center coordinates
            cX, cY = r.center
            pitch = -((cY - 360) / 720) * 45  # Assuming 45 degrees vertical FOV
            yaw = ((cX - 640) / 1280) * 60    # Assuming 60 degrees horizontal FOV
            pitch_yaw_list.append((pitch, yaw))
        return pitch_yaw_list
    
    def get_areas(self):
        areas = []
        for r in self.results:
            corners = np.array(r.corners, dtype=np.float32)
            area = cv2.contourArea(corners)
            areas.append(area)
        return areas

    def get_pixels(self):
        pixels = []
        for r in self.results:
            cX, cY = r.center
            pixels.append((cX, cY))
        return pixels
    
    def get_skews(self):
        skews = []
        for r in self.results:
            corners = np.array(r.corners, dtype=np.float32)
            vector1 = corners[1] - corners[0]
            vector2 = corners[2] - corners[1]
            angle1 = np.arctan2(vector1[1], vector1[0])
            angle2 = np.arctan2(vector2[1], vector2[0])
            skew = np.degrees(angle2 - angle1)
            skews.append(skew)
        return skews

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
        json_str = json.dumps(data, default=_json_default)
        self.pub.set(json_str)

def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


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
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        results = detector.detect(gray)
        tag_data['hasTargets'] = len(results) > 0
        for result in results:
            tag_id = int(result.tag_id)
            center = [float(result.center[0]), float(result.center[1])]
            corners = [[float(c[0]), float(c[1])] for c in result.corners]
            pitch, yaw = detector.get_pitch_yaw()[0] if detector.get_pitch_yaw() else (0, 0)
            area = detector.get_areas()[0] if detector.get_areas() else 0
            pixels = detector.get_pixels()[0] if detector.get_pixels() else (0, 0)
            skew = detector.get_skews()[0] if detector.get_skews() else 0
            tag_data[tag_id] = {
                "center": center,
                "corners": corners,
                "targetPitch": pitch,
                "targetYaw": yaw,
                "targetArea": area,
                "targetPixelsX": pixels[0],
                "targetPixelsY": pixels[1],
                "targetSkew": skew
            }
            string_pub.set(tag_data)

        # Draw boxes on frame
        frame = detector.draw_boxes(output_img)

        # Write image to UI
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ok:
            img_pub.set(jpg.tobytes())

        # Write image to output stream
        output_stream.putFrame(output_img)

        time.sleep(1/30)  # Simulate 30 FPS


if __name__ == "__main__":
    main()
