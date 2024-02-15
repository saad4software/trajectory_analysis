import numpy as np
import cv2
from dense_flow import dense_flow
from tracker import Tracker
from util import *
from model import create_model


file_name="BIRD_011"
cap = cv2.VideoCapture(f"V_{file_name}.mp4")

ret, frame1 = cap.read()

fps = cap.get(cv2.CAP_PROP_FPS)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), int(fps), (int(width), int(height)) )

# check if the file exists
if not ret:
    print("invalid file")

counter = 0

# kalman filter tracker
tracker = Tracker(50, 5, 5, 0)

noise_model = create_model(2, "target_noise_model_v2_10p_96.keras")
classification_model = create_model(3, "target_classes_model_30p_90.keras")


trajectories_dict = {}

while True:

    ret, frame2 = cap.read()
    if not ret:
        break

    counter += 1

    # get dense optical flow boxes
    dense_image, bboxes = dense_flow(frame1, frame2)

    # convert boxes to center points
    centers = boxes2centers(bboxes)

    tracker.update(centers)

    dense_image, trajectories_dict = draw_tracks(
        dense_image, 
        tracker, 
        trajectories_dict=trajectories_dict, 
        model=noise_model,
        binary=True)

    # dense_image, trajectories_dict = draw_tracks(
    #     dense_image, 
    #     tracker, 
    #     trajectories_dict=trajectories_dict, 
    #     model=classification_model,
    #     binary=False)

    cv2.imshow('Original', dense_image)
    writer.write(dense_image)


    key = cv2.waitKey(10)
    if key == 27:
        break  # ESC key pressed

    frame1 = frame2



writer.release()
cap.release()
cv2.destroyAllWindows()