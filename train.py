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
data = np.array(data)
(train_x, test_x, train_y, test_y) = train_test_split(data, test_size=0.20, random_state=42)

