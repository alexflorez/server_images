# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


def PredictTop5_VGG19(image_path):
    model = VGG19()
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    del model
    K.clear_session()
    labels = decode_predictions(preds, top=5 )
    labels = np.array(labels)
    labels = np.squeeze(labels)
    labels = labels[:,1:3]
    for i in range(labels.shape[0]):
        labels[i,0] = str(i+1) + ' - ' + labels[i,0]
    labels[:,1] = np.round(labels[:,1].astype(np.float) * 100, decimals=2)
    labels[:,1] = labels[:,1].astype(np.chararray) + '%'
    labels = dict(labels)
    print (labels)
    return labels
    


def PredictTop5(image_path, model_name):
    models = { 'VGG19': PredictTop5_VGG19 }
    return models[model_name](image_path) 
    
    