import glob
import os
import cv2
import time
import face_detection
import numpy as np

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


if __name__ == "__main__":
    # DSFDDetector
    # RetinaNetMobileNetV1
    # RetinaNetResNet50
    detector = face_detection.build_detector("RetinaNetResNet50",max_resolution=1080)
    
    cap = cv2.VideoCapture('0')

    while(True):
       
        ret, frame = cap.read()
        t = time.time()
        dets = detector.detect(frame[:, :, ::-1])[:, :4]
        print(f"Detection time: {time.time()- t:.3f}")
        draw_faces(frame, dets)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()