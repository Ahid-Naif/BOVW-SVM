import cv2
import numpy as np

class DetectKeypoints:
    def detectSIFT(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        SIFT = cv2.xfeatures2d.SIFT_create() # initialize SIFT detector
        SIFTkeyPoints = SIFT.detect(gray, None) # detect keypoints

        return SIFTkeyPoints