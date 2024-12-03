import sys
sys.path.append("../")
from utils import read_video, save_video
import numpy as np
import cv2 as cv
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator

def main():
    video_frames = read_video(r"A:\ProgrmmingStuff\Football-Analysis\input_videos\08fd33_4.mp4")
    
    tracker = Tracker(r"A:\ProgrmmingStuff\Football-Analysis\models\best.pt")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=r"A:\ProgrmmingStuff\Football-Analysis\stubs\track_stubs.pkl")
    
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=r"A:\ProgrmmingStuff\Football-Analysis\stubs\camera_movement_stub.pkl"
    )
    
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track["box"],
                player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]
    
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_box = tracks["ball"][frame_num][1]["box"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_box)
        
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    
    video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    video_frames = camera_movement_estimator.get_camera_movement(video_frames, camera_movement_per_frame)
    
    save_video(video_frames, r"A:\ProgrmmingStuff\Football-Analysis\output_videos\output.avi")
     
if __name__ == "__main__":
    main()
