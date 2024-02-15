import numpy as np
import cv2

def is_inside(point, rect):
    return rect[0]<point[0]<rect[0]+rect[2] and rect[1]<point[1]<rect[1]+rect[3]


def boxes2centers(bboxes):
    centers = []
    for box_index, box in enumerate(bboxes):
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2
        center = [[center_x], [center_y]]
        centers += [center]

    return centers


def draw_tracks(frame, tracker, trajectories_dict={}, model=None, binary=True):
    for track in tracker.tracks:
        if not track.trace: break

        for j in range(len(track.trace) - 1):
            # Draw trace line
            x1 = track.trace[j][0][0]
            y1 = track.trace[j][1][0]
            x2 = track.trace[j + 1][0][0]
            y2 = track.trace[j + 1][1][0]

            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))

            # cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                
        trace_x = track.trace[-1][0][0]
        trace_y = track.trace[-1][1][0]
        
        # update trajectory info
        track_data = [trace_x/frame.shape[1], trace_y/frame.shape[0]] 
        track_id = str(track.track_id)
        if track_id not in trajectories_dict:
            trajectories_dict[track_id] = [(track_data)]
        else:
            trajectories_dict[track_id] += [(track_data)]

        color = (0, 0, 255) if len(trajectories_dict[track_id]) > 10 else (255, 255, 255)

        target_class = "unknown"

        if binary and len(trajectories_dict[track_id]) > 10 and model:
            prediction = model.predict(np.array([trajectories_dict[track_id][-10:]]))
            print(prediction)
            prediction = prediction[0] == max(prediction[0])
            target_class = "target" if prediction[0] else "noise"

        elif not binary and len(trajectories_dict[track_id]) > 30 and model:
            prediction = model.predict(np.array([trajectories_dict[track_id][-30:]]))
            print(prediction)
            prediction = prediction[0] == max(prediction[0])
            target_class = "airplane" if prediction[0] else "bird" if prediction[1] else "drone"


        # cv2.putText(frame, f'ID: {track.track_id}', (int(trace_x), int(trace_y)),  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        cv2.putText(frame, target_class, (int(trace_x), int(trace_y)),  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

    return frame, trajectories_dict


def draw_points(frame, points):
    for pt in points:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 0, [255, 0, 0], 2)
    return frame