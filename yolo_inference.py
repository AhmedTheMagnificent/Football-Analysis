from ultralytics import YOLO

model = YOLO(r"A:\ProgrmmingStuff\Football-Analysis\models\best.pt")

results = model.predict(r"A:\ProgrmmingStuff\Football-Analysis\input_videos\08fd33_4.mp4", save=True)
print(results[0])
print("--------------------------------------------")
for box in results[0].boxes:
    print(box)