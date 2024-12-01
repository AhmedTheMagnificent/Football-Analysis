import cv2 as cv

def read_video(video_path):
    cap = cv.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, video_path):
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter(video_path, fourcc, 24.0, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()