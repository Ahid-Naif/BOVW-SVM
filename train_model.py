from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
import numpy as np
import pickle
import h5py
import cv2

datasetPath = "caltech5_dataset" # dataset path
dbPath = "database/features.hdf5" # database path
bovw_db = "database/bovw.hdf5" # bovw database path
modelStoragePath = "database/model.p" # model path

# open the features and bag-of-visual-words databases
featuresDB = h5py.File(dbPath)
bovwDB = h5py.File(bovw_db)

# grab the training and testing data from the dataset using the first 300
# images as training and the remaining 200 images for testing
print("[INFO] loading data...")
(trainData, trainLabels) = (bovwDB["bovw"][:300], featuresDB["image_ids"][:300])
(testData, testLabels) = (bovwDB["bovw"][300:], featuresDB["image_ids"][300:])
print(trainLabels)
# prepare the labels by removing the filename from the image ID, leaving
# us with just the class name
trainLabels = [label.split(":")[0] for label in trainLabels]
testLabels = [label.split(":")[0] for label in testLabels]

# define the grid of parameters to explore, then start the grid search where
# we evaluate a Linear SVM for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LinearSVC(random_state=42), params, cv=3)
model.fit(trainData, trainLabels)
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# show a classification report
print("[INFO] evaluating...")
predictions = model.predict(testData)

print("[INFO] Confusion Matrix:")
print(confusion_matrix(testLabels, predictions))
print("[INFO] Classification Report:")
print(classification_report(testLabels, predictions))

# close the databases
featuresDB.close()
bovwDB.close()

# dump the classifier to file
print("[INFO] dumping classifier to file...")
pickle.dump(model, open(modelStoragePath, "wb"))