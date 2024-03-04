import numpy as np
import cv2
import os
import time
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import joblib
import tracemalloc

# Start tracing memory allocations
tracemalloc.start()

def get_label(img_path):
    # Extract the label from the filename
    filename = os.path.basename(img_path)
    label = filename.split("_")[0]
    return label

start_time = time.time()
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
            X_train.append(img)
            Y_train.append(get_label(img_path))
# Convert to numpy array
X_train = np.array(X_train)
Y_train = np.array(Y_train)

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
            X_test.append(img)
            Y_test.append(get_label(img_path))

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Print the shapes of the arrays
print('total images: ', total)
print("Xtrain shape:", X_train.shape)
print("Ytrain shape:", Y_train.shape)
print("Xtest shape:", X_test.shape)
print("Ytest shape:", Y_test.shape)
# Shuffle the dataset
X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
# Split the dataset
#Reshape sets
X_train, X_test = X_train.reshape(-1,32*32*3), X_test.reshape(-1,32*32*3)
Y_train, Y_test = Y_train.flatten(), Y_test.flatten()
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# Initialize the model
# An SGDClassifier with these parameters would resemble an implementation of an ISVM but not an SVM
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1)

# Load the existing model if it exists
try:
    clf = joblib.load('Z:\ProposalAlgos\ISVM\ISVM\ISVMmodel.joblib')
    print('Loaded existing model')
except:
    print('No existing model found')

#Train in Epochs since partial_fit doesn't use max_iter
n_epochs = 100
# Split the dataset into batches
batch_size = 128
n_batches = int(np.ceil(len(X_train) / batch_size))
clf.partial_fit(X_train[:1], Y_train[:1], classes=np.unique(Y_train))
# Train the model 
for epoch in range(n_epochs):
    print("Training epoch:", epoch + 1)
    for i in range(n_batches):
        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_batch = Y_train[i*batch_size:(i+1)*batch_size]
        clf.partial_fit(X_batch, y_batch, classes=np.unique(Y_train))

joblib.dump(clf, 'Z:\ProposalAlgos\ISVM\ISVM\ISVMmodel.joblib')
print("saved model: ISVMmodel.joblib")

Y_test_preds = clf.predict(X_test)
Y_train_preds = clf.predict(X_train)

#results
print("Train Accuracy: {}".format(accuracy_score(Y_train, Y_train_preds)))
print("Test Accuracy: {}".format(accuracy_score(Y_test, Y_test_preds)))

#model details
print('Coefficients shape:', clf.coef_.shape)

#perfomance measures
Running_time = time.time() - start_time
print("Running time: %.2f seconds" % Running_time)



# Get the peak memory usage
_ , mem_usage = tracemalloc.get_traced_memory()
print(tracemalloc.get_traced_memory())
print(f"Peak memory usage: {mem_usage / 10**6} MB")
tracemalloc.stop()