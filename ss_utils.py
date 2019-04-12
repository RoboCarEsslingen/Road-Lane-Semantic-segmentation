import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import itertools
from glob import glob
import re
from keras.models import *
from keras.layers import *
from keras import models
from keras import layers
from keras.utils import plot_model
import random
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from keras import optimizers
import matplotlib.pyplot as plt

def fcn_predict(fcn_model,raw_image,n_Classes,input_height,input_width,cols=[(0,0,0),(0,255,0)]):
  im_bgd=cv2.resize(raw_image,(input_width,input_height))
  X = im_bgd.astype(np.float32)
  X = X/255.0
  pr = fcn_model.predict(np.array([X]))
  pr = pr.reshape((input_height,input_width, n_Classes)).argmax( axis=2 )
  im_pred = np.zeros((input_height,input_width, 3))
  for c in range(n_Classes):
    im_pred[:,:,0] += ((pr[:,:] == c )*(cols[c][0])).astype('uint8')
    im_pred[:,:,1] += ((pr[:,:] == c )*(cols[c][1])).astype('uint8')
    im_pred[:,:,2] += ((pr[:,:] == c )*(cols[c][2])).astype('uint8')
  im_pred = cv2.resize(im_pred,(input_width,input_height))
  im_pred=np.asarray(im_pred,np.uint8)
  return im_bgd,im_pred

def get_data(data_folder, n_classes, input_height, input_width):
  image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
  label_paths = {
      re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
      for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
  X = []
  Y = []
  for image in image_paths:
    im = cv2.imread(image, 1)
    im = cv2.resize(im,(input_width, input_height))
    im = im.astype(np.float32)
    im = im/255.0
    X.append(im)
    
    seg_labels = np.zeros((input_height, input_width, n_classes))
    segpath = label_paths[os.path.basename(image)]
    seg = cv2.imread(segpath, 1)
    seg = cv2.resize(seg,(input_width, input_height))
    seg = seg[:, : , 0]
    for c in range(n_classes):
      if c==0:
        seg_labels[: , : , c ] = (seg == c).astype(int)
      else:
        seg_labels[: , : , c ] = (seg > 0 ).astype(int)
    seg_labels = np.reshape(seg_labels, (input_height*input_width, n_classes))
    Y.append(seg_labels)
  return np.asarray(X),np.asarray(Y)

def augment_training_data(train_im,train_lab,n_classes,input_height,input_width):
  augmented_lab=[]
  augmented_im=[]
  for i in range(len(train_im)):
    im=train_im[i,:,:,:].reshape(input_height, input_width, 3)
    augmented_im.append(im)
    augmented_im.append(cv2.flip(im,1))
    lab=train_lab[i,:,:].reshape(input_height,input_width,n_classes)
    augmented_lab.append(np.reshape(lab,(input_height*input_width,n_classes)))
    augmented_lab.append(np.reshape(cv2.flip(lab,1),(input_height*input_width,n_classes)))
  l = list(zip(augmented_lab,augmented_im))
  random.shuffle(l)
  augmented_lab,augmented_im=zip(*l)
  augmented_im=np.asarray(augmented_im)
  augmented_lab=np.asarray(augmented_lab)
  return augmented_lab,augmented_im

def fcn32(n_Classes,topless_weights_path,input_height,input_width):
  img_input = Input(shape=(input_height,input_width,3))
  
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_initializer='he_normal')(img_input)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_initializer='he_normal')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
  
  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',kernel_initializer='he_normal')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',kernel_initializer='he_normal')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
  
  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',kernel_initializer='he_normal')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',kernel_initializer='he_normal')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',kernel_initializer='he_normal')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
  
  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',kernel_initializer='he_normal')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',kernel_initializer='he_normal')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',kernel_initializer='he_normal')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
  
  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',kernel_initializer='he_normal')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',kernel_initializer='he_normal')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',kernel_initializer='he_normal')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
  
  vgg  = Model(img_input , x)
  vgg.load_weights(topless_weights_path)
  
  x = (Conv2D(4096 , (7 , 7 ) , activation='relu' , padding='same',kernel_initializer='he_normal',name="conv6"))(x)
  x = Dropout(0.5)(x)
  x = (Conv2D(4096 , (1 , 1 ) , activation='relu' , padding='same',kernel_initializer='he_normal',name="conv7"))(x)
  x = Dropout(0.5)(x)
  
  x = (Conv2D(n_Classes ,  ( 1 , 1 ),padding='same' ,kernel_initializer='he_normal',name="scorer1"))(x)
  x = Conv2DTranspose(n_Classes , kernel_size=(64,64) ,padding='same',  strides=(32,32),name="Upsample32")(x)
  
  o_shape = Model(img_input,x).output_shape
  output_Height = o_shape[1]
  output_Width = o_shape[2]
  
  x = (Reshape((-1  , output_Height*output_Width)))(x)
  x = (Permute((2, 1)))(x)
  x = (Activation('softmax'))(x)
  
  model = Model( img_input , x)
  model.outputWidth = output_Width
  model.outputHeight = output_Height
  for l in vgg.layers:
    if "input" not in l.name:
      w=l.get_weights()
      model.get_layer(l.name).set_weights(w)
      model.get_layer(l.name).trainable=False
    
  return  model

def fcn8(fcnmodel,n_classes):
  x=fcnmodel.get_layer('scorer1').output
#   x=Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(x)
  x=Conv2DTranspose(n_classes, kernel_size=(4,4),kernel_initializer='he_normal',padding='same', strides=(2,2),name="deconv1")(x)
  
  l4 = fcnmodel.get_layer('block4_pool').output
  l4= (Conv2D(n_classes, (1 , 1 ),padding='same',kernel_initializer='he_normal',name="scorer2"))(l4)
  x,l4 = crop(x, l4 , fcnmodel.input)
  x=Add()([l4,x])
#   x = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(x)
  x = Conv2DTranspose(n_classes, kernel_size=(4,4),kernel_initializer='he_normal',padding='same', strides=(2,2),name="deconv2")(x)

  l3 = fcnmodel.get_layer('block3_pool').output
  l3 = (Conv2D(n_classes ,  ( 1 , 1 ) ,padding='same',kernel_initializer='he_normal',name="scorer3"))(l3)
  l3,x = crop(l3,x, fcnmodel.input)
  x  = Add()([l3,x])
#   x = Conv2DTranspose(n_classes , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False)(x)
  x = Conv2DTranspose(n_classes,padding='same',kernel_initializer='he_normal' , kernel_size=(16,16) ,strides=(8,8),name="Upsample8")(x)
  
  o_shape = Model(fcnmodel.input , x).output_shape
  oH = o_shape[1]
  oW = o_shape[2]
  x = (Reshape((-1  , oH*oW)))(x)
  x = (Permute((2, 1)))(x)
  x = (Activation('softmax'))(x)
  M = Model(input=fcnmodel.input, output=x)	
  for l in M.layers:
    l.trainable=True
  M.outputWidth = oW
  M.outputHeight = oH
  return M

def crop(a,b,input_image):
  
  o_shape1 = Model(input_image, a ).output_shape
  outputHeight1 = o_shape1[1]
  outputWidth1 = o_shape1[2]

  o_shape2 = Model(input_image , b).output_shape
  outputHeight2 = o_shape2[1]
  outputWidth2 = o_shape2[2]
    
  cx = abs( outputWidth1 - outputWidth2 )
  cy = abs( outputHeight2 - outputHeight1 )
  
  if outputWidth1 > outputWidth2:
    a = Cropping2D(cropping=((0,0) , ( 0 , cx )))(a)
  else:
    b = Cropping2D(cropping=((0,0) , ( 0 , cx )))(b)
    
  if outputHeight1 > outputHeight2 :
    a = Cropping2D(cropping=((0,cy) , ( 0 , 0 )))(a)
  else:
    b = Cropping2D(cropping=((0, cy ) , ( 0 , 0)))(b)
  return a , b
  
def save_file_to_drive(drive_serv,name, path):
  file_metadata = {'name': name,'mimeType': 'application/octet-stream'}
  media = MediaFileUpload(path, mimetype='application/octet-stream',resumable=True)
  created = drive_serv.files().create(body=file_metadata,media_body=media,fields='id').execute()
  print('File ID: {}'.format(created.get('id')))
  return created
