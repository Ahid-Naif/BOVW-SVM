import cv2

class ComputeKeypoints:
    def computeSIFT(self, image, keypoints):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        SIFT = cv2.xfeatures2d.SIFT_create()
        kps, des = SIFT.compute(gray, keypoints) # compute keypoints & their descriptions

        return kps, des