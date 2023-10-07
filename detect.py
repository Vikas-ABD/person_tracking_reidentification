import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO
from tracker_DL import *
from tracker_cv import *

# Create a directory to store cropped images
#os.makedirs("detected_images", exist_ok=True)

model = YOLO('best.pt')
cap = cv2.VideoCapture('video2.mp4')

my_file = open("label.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()
trackertwo = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    frame = cv2.resize(frame, (640, 640))
    results = model.predict(frame)
    a = results[0].boxes.xyxy
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        list.append([x1, y1, x2, y2])

    bbox_idx = trackertwo.update(list, frame)
    
    for bbox in bbox_idx:
        x1, y1, x2, y2, id = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        
        # Crop the detected person from the frame
        #person_roi = frame[y1:y2, x1:x2]
        
        # Save the cropped person image to the "detected_images" folder
        ##image_filename = os.path.join("detected_images", f"person_{id}.jpg")
        #cv2.imwrite(image_filename, person_roi)
    
    cv2.imshow("FRAME", frame)
    
    # Check for 'q' key press to stop the video display
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
