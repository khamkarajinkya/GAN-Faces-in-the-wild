import os
from random import randint
import random
from PIL import Image,ImageOps
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Convolution2D, Dense, Dropout, Activation,Flatten,Reshape,BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose
import matplotlib.pyplot as plt
import numpy as np
import re
from IPython import display
from tqdm import tqdm



images = np.ndarray(shape=(1521,40,40,1),dtype=np.float32)

i=0

for root, dirnames, filenames in os.walk("..."):
    for filename in filenames:
        if re.search("\.(pgm)$", filename):
            filepath = os.path.join(root, filename)
            head, tail = os.path.split(filepath)
            image = (load_img(str(tail.split(".")[0]+".pgm")))
            image=image.resize((40,40),Image.ANTIALIAS)
            image=ImageOps.grayscale(image)
            image=img_to_array(image)
            r,c,ch=image.shape
            images[i]=image.reshape(40,40,1)
            i += 1
            if i % 250 == 0:
                print("%d images to array" % i)
    print("All images to array!")
    
for i in range (0,10):
    fig,ax = plt.subplots(1)

    # Display the image
    index=randint(0,1500)
    ax.imshow(images[index].reshape(40,40),cmap='gray')
    plt.show()
    
images = images/255.0

#Generator 

input_gen = Input (shape = [100]) 

d1 = Dense(256*10*10)(input_gen)
b1 = BatchNormalization (momentum = 0.9, center = True , scale = True)(d1)
a1 = Activation ('relu')(b1)
r1 = Reshape ((10,10,256))(a1)
u1 = UpSampling2D()(r1)
c1 = Conv2DTranspose (128,5,padding = 'same')(u1)
b2 = BatchNormalization (momentum = 0.9, center = True , scale = True)(c1)
a2 = Activation ('relu')(b2)
u2 = UpSampling2D()(a2)
c2 = Conv2DTranspose (64,5,padding = 'same')(u2)
b3 = BatchNormalization (momentum = 0.9, center = True , scale = True)(c2)
a3 = Activation ('relu')(b3)
c3 = Conv2DTranspose (32,5,padding = 'same')(a3)
b4 = BatchNormalization (momentum = 0.9, center = True , scale = True)(c3)
a4 = Activation ('relu')(b4)
c4 = Conv2DTranspose (1,5,padding = 'same')(a4)
a5 = Activation ('tanh')(c4)
gen = Model (inputs = input_gen, outputs = a5)
gen.compile(loss='binary_crossentropy', optimizer="adam")

gen.summary ()

#discriminator 

input_des = Input (shape = (40,40,1))
c1 = Convolution2D(64, 5, strides = 2, activation = LeakyReLU(0.2), padding = 'same')(input_des)
d1 = Dropout (0.4)(c1)
c2 = Convolution2D(128, 5, strides = 2, activation = LeakyReLU(0.2), padding = 'same')(d1)
d2 = Dropout (0.4)(c2)
c3 = Convolution2D(256, 5, strides = 1, activation = LeakyReLU(0.2), padding = 'same')(d2)
d3 = Dropout (0.4)(c3)
f1 = Flatten()(d3)
de1 = Dense (128)(f1)
d4 = Dropout (0.4)(de1)
de2 = Dense (2,activation = 'softmax')(d4)

dis = Model (inputs = input_des , outputs = de2)
dis.compile(loss='categorical_crossentropy', optimizer="adam")


dis.summary()

dis.trainable = False
