from utils import read_video, save_video
from trackers import Tracker
import cv2 as cv

def main():
    video_frames = read_video(r"A:\ProgrmmingStuff\Football-Analysis\input_videos\08fd33_4.mp4")
    
    tracker = Tracker(r"A:\ProgrmmingStuff\Football-Analysis\models\best.pt")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=r"A:\ProgrmmingStuff\Football-Analysis\stubs\track_stubs.pkl")
    
    
    video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(video_frames, r"A:\ProgrmmingStuff\Football-Analysis\output_videos\output.avi")
     
if __name__ == "__main__":
    main()
