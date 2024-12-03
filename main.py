import sys
import os
sys.path.append("../")  # Ensure that the path is correctly pointing to the parent directory
from utils import read_video, save_video
import numpy as np
import cv2 as cv
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator

def main():
    video_path = r"A:\ProgrmmingStuff\Football-Analysis\input_videos\08fd33_4.mp4"
    video_frames = read_video(video_path)
    
    # Ensure the video frames are loaded correctly
    if video_frames is None or len(video_frames) == 0:
        print(f"Failed to load video from {video_path}")
        return
    
    # Initialize the tracker and get object tracks
    tracker = Tracker(r"A:\ProgrmmingStuff\Football-Analysis\models\best.pt")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=r"A:\ProgrmmingStuff\Football-Analysis\stubs\track_stubs.pkl")
    
    tracker.add_position_to_tracks(tracks)
    
    # Check if tracks were successfully loaded
    if not tracks:
        print("Failed to load tracks.")
        return
    
    # Estimate camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=r"A:\ProgrmmingStuff\Football-Analysis\stubs\camera_movement_stub.pkl"
    )
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Assign teams and team colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Assign team to each player
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track["box"],
                player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]
    
    # Assign ball to the player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_box = tracks["ball"][frame_num][1]["box"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_box)
        
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            # If no player is assigned, keep the previous ball control
            team_ball_control.append(team_ball_control[-1] if team_ball_control else None)
    
    team_ball_control = np.array(team_ball_control)
    
    # Draw annotations on video frames
    video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    # Overlay camera movement on frames
    video_frames = camera_movement_estimator.draw_camera_movement(video_frames, camera_movement_per_frame)
    
    # Save the final video with annotations
    output_path = r"A:\ProgrmmingStuff\Football-Analysis\output_videos\output.avi"
    save_video(video_frames, output_path)
    print(f"Output video saved to {output_path}")
     
if __name__ == "__main__":
    main()
