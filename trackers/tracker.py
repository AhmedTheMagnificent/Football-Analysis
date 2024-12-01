from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import cv2 as cv
import os
import sys
sys.path.append("../")
from utils import get_box_width, get_center_of_box


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as file:
                tracks = pickle.load(file)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {value: key for key, value in cls_names.items()}

            print(f"Frame {frame_num}: Detected classes: {list(cls_names.values())}")

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names.get(class_id) == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_names_inv.get("player", None)

            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detections_with_tracks:
                box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv.get("player", None):
                    tracks["players"][frame_num][track_id] = {"box": box}

                if cls_id == cls_names_inv.get("referee", None):
                    tracks["referees"][frame_num][track_id] = {"box": box}

            for frame_detection in detection_supervision:
                box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv.get("ball", None):
                    tracks["ball"][frame_num][1] = {"box": box}

        if stub_path is not None:
            with open(stub_path, "wb") as file:
                pickle.dump(tracks, file)

        return tracks

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections += self.model.predict(frames[i:i + batch_size], conf=0.1)
        return detections

    def draw_ellipse(self, frame, bbox, color, track_id=None, draw_track_id=True):
        y2 = int(bbox[3])  # Bottom of the bounding box
        x_center, _ = get_center_of_box(bbox)  # Use the correct method
        width = get_box_width(bbox)  # Use the correct method

        # Draw ellipse
        cv.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv.LINE_4,
        )

        # Skip rectangle and text for objects where track_id is not required
        if not draw_track_id or track_id is None:
            return frame

        # Draw rectangle for track ID
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = y2 - rectangle_height // 2 + 15
        y2_rect = y2 + rectangle_height // 2 + 15

        cv.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv.FILLED)

        # Adjust text position for multi-digit IDs
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        # Draw track ID text
        cv.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        return frame
    

    def draw_triangle(self, frame, box, color):
        y = int(box[1])
        x, _ = get_center_of_box(box)
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv.drawContours(frame,
                        [triangle_points],
                        0,
                        color,
                        -1)
        cv.drawContours(frame,
                        [triangle_points],
                        0,
                        (0, 0, 0),
                        2)
        return frame


    def draw_annotations(self, input_video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(input_video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players with track ID
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["box"], (0, 0, 255), track_id, draw_track_id=True)

            # Draw referees without track ID
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["box"], (0, 255, 255), draw_track_id=False)

            # Draw the ball without track ID
            
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["box"], (0, 255, 0))
            
            output_video_frames.append(frame)

        return output_video_frames
