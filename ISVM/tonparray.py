import numpy as np
import cv2
import os
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

def get_label(img_path):
    # Extract the label from the filename
    filename = os.path.basename(img_path)
    label = filename.split("_")[0]
    return label

# Load images
path = "Z:\ProposalAlgos\humandetectiondataset\\train"
total = 0
X_train, Y_train = [], []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".png"):
            total = total+1 
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            #fd, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
            X_train.append(img)
            Y_train.append(get_label(img_path))
# Convert to numpy array
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
np.save('Z:\ProposalAlgos\ISVM\ISVM\X_train.npy', X_train)
np.save('Z:\ProposalAlgos\ISVM\ISVM\Y_train.npy', Y_train)

#Repeat for test set
X_test, Y_test = [], []
path = "Z:\ProposalAlgos\humandetectiondataset\\test"
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".png"):
            total = total+1 
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            #fd, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
            X_test.append(img)
            Y_test.append(get_label(img_path))
label_encoder = LabelEncoder()
Y_test = label_encoder.fit_transform(Y_test)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
np.save('Z:\ProposalAlgos\ISVM\ISVM\X_test.npy', X_test)
np.save('Z:\ProposalAlgos\ISVM\ISVM\Y_test.npy', Y_test)