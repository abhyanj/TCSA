#VideoProcessing

#packages
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

#arguments
arg = argparse.ArgumentParser()
arg.add_argument("-m", "--model", required = True, help = "path to trained model")
arg.add_argument("-l", "--label-bin", required = True, help = "path to libel binarizer")
arg.add_argument("-i", "--input", required = True, help = "path to video")
arg.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")


