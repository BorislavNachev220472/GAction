# Description: This file contains the functions that are used to calculate the metrics and to predict the image using the model.
import tkinter as tk
from tkinter import filedialog, Canvas, NW, Toplevel
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import tensorflow as tf
import cv2
import keras.backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
from patchify import patchify,unpatchify

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        threshold = 0.5 
        y_pred_binary = K.round(y_pred + 0.5 - threshold)
        
        intersection = K.sum(K.abs(y_true * y_pred_binary), axis=[1,2,3])
        total = K.sum(K.square(y_true), [1,2,3]) + K.sum(K.square(y_pred_binary), [1,2,3])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())
    
    return K.mean(f(y_true, y_pred), axis=-1)

def padder(image,filename, patch_size):
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE)
    return padded_image

def predict_image(image, model):
    image = padder(image,'none', 256)
    patches = patchify(image, 256, 256)
    x = patches.shape[0]
    y = patches.shape[1]
    patches = patches.reshape(-1,256,256)
    predictions = model.predict(patches,verbose=0)
    patches = predictions.reshape(x, y, 256, 256)
    predicted_image = unpatchify(patches,image.shape)
    return predicted_image