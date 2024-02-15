import numpy as np
import cv2

def dense_flow(original_frame1, original_frame2, pyramid=0, min_dim=15, max_dim=60):
    frame1 = original_frame1.copy()
    frame2 = original_frame2.copy()

    bboxes = []
    if pyramid > 0:
        for _ in range(pyramid):
            frame1 = cv2.pyrDown(frame1)
            frame2 = cv2.pyrDown(frame2)

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    opticalFlow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 1, 20, 3, 5, 1.2, 0)
    originalMag, originalAng = cv2.cartToPolar(opticalFlow[..., 0], opticalFlow[..., 1])

    originalMag = cv2.normalize(originalMag, None, 0, 255, cv2.NORM_MINMAX)
    originalAng = originalAng * 180 / np.pi / 2
    
    ret, magMask = cv2.threshold(originalMag, 150, 255, cv2.THRESH_BINARY)
    
    if pyramid > 0:
        for _ in range(pyramid):
            magMask = cv2.pyrUp(magMask)
    
    magMask = np.uint8(magMask)
    contours, hierarchy = cv2.findContours(magMask, 1, 2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        multiplier = pyramid + 1
        if( min_dim*multiplier < h < max_dim*multiplier and min_dim*multiplier < w < max_dim*multiplier ):
            center = np.array([[x + w / 2], [y + h / 2]])
            # centers.append(np.round(center))
            bbox = (x, y, w, h)
            bboxes.append(bbox)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            original_frame1 = cv2.rectangle(original_frame1, p1, p2, (255, 0, 0), 1)
    
    return original_frame1, bboxes