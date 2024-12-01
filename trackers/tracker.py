from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import cv2 as cv
import pandas as pd
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
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("box",[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"box":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frame = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frame = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frame / (team_1_num_frame + team_2_num_frame)
        team_2 = team_2_num_frame / (team_1_num_frame + team_2_num_frame)
        
        cv.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame

    def draw_annotations(self, input_video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(input_video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players with track ID
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["box"], color, track_id, draw_track_id=True)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["box"], (0, 0, 255))

            # Draw referees without track ID
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["box"], (0, 255, 255), draw_track_id=False)

            # Draw the ball without track ID
            
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["box"], (0, 255, 0))
            
            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)

        return output_video_frames
