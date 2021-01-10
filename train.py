#training file
import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D #resnet50 CNN
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split # used this before 
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os # file stuff
# all the stuff we need

#from VideoProcessing.py (might be easier to keep within this file but let's see)
arg = argparse.ArgumentParser()
arg.add_argument("-d", "--dataset", required = True, help = "path to our TCSA data")
arg.add_argument("-m", "--model", required = True, help = "path to trained model")
arg.add_argument("-l", "--label-bin", required = True, help = "path to libel binarizer")
arg.add_argument("-i", "--input", required = True, help = "path to video")
arg.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(arg.parse_args())
#now actually getting the data. We only have TCSA so no need to label or check anything
imagePaths = list(paths.list_images(args["dataset"]))
data = []
#not needed: labels = []

for i in imagePaths:
	# load the image, convert it to RGB channel ordering, and resize
	# it to be a fixed 224x224 pixels, ignoring aspect ratio
	image = cv2.imread(i)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	# update the data and labels lists, respectively
	data.append(image)
	#don't think we need: labels.append(label)

#gotta check the one-hot encoding stuff

data = np.array(data)
(train_x, test_x, train_y, test_y) = train_test_split(data, test_size=0.20, random_state=42)

#train data object initialization
trainAug = ImageDataGenerator(rotation_range=30, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order). Set the mean construction value
mean = np.array([123.68, 116.779, 103.939], dtype="float32") #numbers are from ImageNet
trainAug.mean = mean # for mean subtraction 
valAug.mean = mean 


# load the ResNet-50 network, ensuring the head FC layer sets are left off 
baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False






