# How to run:
# python train_network.py --dataset insects_images --model insects.model
# --------------------------------------------------------------------------
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
#from pyimagesearch.lenet import LeNet
# cnkhanh
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
# cnkhanh
from os import listdir
import json




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
  help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
  help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
  help="path to output loss/accuracy plot")
# cnkhanh
ap.add_argument("--number_epochs", type=int, default=25,
  help="number of epochs")  
ap.add_argument("--train_file", type=str, default="train.txt",
  help="path to output train file")  
ap.add_argument("--test_file", type=str, default="test.txt",
  help="path to output test file") 
ap.add_argument("--labels_file", type=str, default="index_labels.txt",
  help="path to output index-labels file") 
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
#EPOCHS = 25
EPOCHS = args["number_epochs"]
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
# cnkhanh
filenames = []
out_train_file = args["train_file"]
out_test_file = args["test_file"]
out_labels_file = args["labels_file"]

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# cnkhanh
class_names = sorted(listdir(args["dataset"]))
# Write labels to files with index
print("Write index_labels.txt files")
with open(out_labels_file, 'w') as f:
    f.write(json.dumps(class_names))

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)
	
	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	#print("label ban dau: ",label)
	#label = 1 if label == "santa" else 0
	# cnkhanh
	label = class_names.index(label)
	#print("image: ",imagePath)
	#print("label: ",label)
	labels.append(label)
	
	# cnkhanh	
	filename = imagePath.split(os.path.sep)[-2]+"-"+imagePath.split(os.path.sep)[-1]
	# cnkhanh	
	filenames.append(filename)
	
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
#(trainX, testX, trainY, testY) = train_test_split(data,
#	labels, test_size=0.25, random_state=42)

# partition the data into training and testing splits using 75% of - cnkhanh 1/4
#(trainX, testX, trainY, testY, trainfiles, testfiles) = train_test_split(data,
#	labels, filenames, train_size=2/3, test_size=1/3)
(trainX, testX, trainY, testY, trainfiles, testfiles) = train_test_split(data,
	labels, filenames, train_size=3/4, test_size=1/4, shuffle=False)

# cnkhanh - write train files and test files
print("Write train.txt files")
out_train= open(out_train_file,"w")
for train_item in sorted(trainfiles):
	out_train.write(train_item.partition('-')[0]+"\t"+train_item.partition('-')[2]+"\n")
out_train.close()
# cnkhanh - write train files and test files
print("Write test.txt files")
out_test= open(out_test_file,"w")
for test_item in sorted(testfiles):
	out_test.write(test_item.partition('-')[0]+"\t"+test_item.partition('-')[2]+"\n")
out_test.close()

# convert the labels from integers to vectors
#trainY = to_categorical(trainY, num_classes=2)
#testY = to_categorical(testY, num_classes=2)

# convert the labels from integers to vectors - cnkhanh
trainY = to_categorical(trainY, num_classes=len(class_names))
testY = to_categorical(testY, num_classes=len(class_names))

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
	
# initialize the model
print("[INFO] compiling model...")
#model = LeNet.build(width=28, height=28, depth=3, classes=2)
# cnkhanh
model = LeNet.build(width=28, height=28, depth=3, classes=len(class_names))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
# cnkhanh
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#import tensorflow as tf
#model.add(tf.keras.layers.Dense(len(class_names), activation='softmax'))	
# train the network
print("[INFO] training network...")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
	
# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")
print(H.history['accuracy'][-1])
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy" + " ( " + str(H.history['accuracy'][-1]) + " )")
plt.xlabel("Epoch # " + str(N))
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
