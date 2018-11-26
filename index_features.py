# import the necessary packages
from myPackage.Image_Detectors.DetectKeypoints import DetectKeypoints
from myPackage.Image_Descriptors.localFeatures.ComputeKeypoints import ComputeKeypoints
from myPackage.indexer.featureindexer import FeatureIndexer
from imutils import paths
import imutils
import random
import numpy as np
import cv2

datasetPath = "caltech5_dataset" # dataset path
dbStoragePath = "database/features.hdf5" # database path
numImages = 500
maxBufferSize = 50000

# initialize the keypoint detector, local invariant descriptor
detector = DetectKeypoints()
descriptor = ComputeKeypoints()

# initialize the feature indexer, then grab the image paths and randomly shuffle them
fi = FeatureIndexer(dbStoragePath, estNumImages=numImages, maxBufferSize=maxBufferSize, verbose=True)
imagesPaths = list(paths.list_images(datasetPath))
random.shuffle(imagesPaths)

# loop over the images in the dataset
for (i, imagePath) in enumerate(imagesPaths):
	# extract the label and image class from the image path and use it to
	# construct the unique image ID
	
	p = imagePath.split("\\")
	imageID = "{}:{}".format(p[1], p[2])

	# load the image and prepare it from description
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(320, image.shape[1]))

	#detect keypoints
	kps = detector.detectSIFT(image)
	
	# describe the image
	(kps, des) = descriptor.computeSIFT(image, kps)

	# convert kps into numpy array 
	kps = np.int([kp.pt for kp in kps])

	# if either the keypoints or descriptors are None, then ignore the image
	if kps is None or des is None:
		continue

	# index the features
	fi.add(imageID, kps, des)

# finish the indexing process
fi.finish()