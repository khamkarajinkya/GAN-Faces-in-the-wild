#Import required libraries
from random import randint
import random
from PIL import Image,ImageOps
from keras.preprocessing.image import img_to_array, load_img,array_to_img
from sklearn.datasets import fetch_olivetti_faces
from keras.models import Model,Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.layers import Input, Convolution2D, Dense, Dropout, Activation,Flatten,Reshape,BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from tqdm import tqdm

#Fetch Olivetti facial dataset from sklearn

data = fetch_olivetti_faces(data_home = '...')

#load data 
img = data['data']
img = img.reshape(400,64,64,1)

images = np.ndarray(shape=(400,32,32,1),dtype=np.float32)

#convert data to 32*32, required for my laptop, I hope to run the 64*64 version on University resources

for i in range (0,400):
    image = array_to_img(img[i])
    image = image.resize((32,32),Image.ANTIALIAS)
    image=ImageOps.grayscale(image)
    images[i] = img_to_array(image).reshape(32,32,1)
    i+=1
    
images = images/255.0

#Generator

#You start with a vector of noise and keep deconvoluting to generate the image, the last layer is a pixelwise implementation
#Generator [Dense [8,8,256,relu] -> Upsample [16,16,256] -> Convo2dTranspose [128,16,16,stride = 5,relu] -> Upsample [32,32,256] ->
#Convo2dTranspose [64,32,32,stride = 5,relu] -> Convo2dTranspose [32,32,32,stride = 5,relu] -> Convo2dTranspose [1,32,32,stride = 5,TANH]  
#Last layer has tanh as per Soumith's gan hacks
#Each batch norm layer is followed by activation as per the original batch norm player, though for subsequent runs, I am going to play 
#around activation and batch norm layers

input_gen = Input (shape = [100]) 

d1 = Dense(256*8*8)(input_gen)
a0= Activation ('relu')(d1)
b1 = BatchNormalization ()(a0)
r1 = Reshape ((8,8,256))(b1)
u1 = UpSampling2D()(r1)
c1 = Conv2DTranspose (128,5,padding = 'same',kernel_initializer='glorot_normal')(u1)
b2 = BatchNormalization ()(c1)
a2 = Activation ('relu')(b2)
u2 = UpSampling2D()(a2)
c2 = Conv2DTranspose (64,5,padding = 'same',kernel_initializer='glorot_normal')(u2)
b3 = BatchNormalization ()(c2)
a3 = Activation ('relu')(b3)
c3 = Conv2DTranspose (32,5,padding = 'same',kernel_initializer='glorot_normal')(a3)
b4 = BatchNormalization ()(c3)
a4 = Activation ('relu')(b4)
c4 = Conv2DTranspose (1,5,padding = 'same',kernel_initializer='glorot_normal')(a4)
a5 = Activation ('tanh')(c4)
b5 = BatchNormalization ()(a5)
gen = Model (inputs = input_gen, outputs = b5)
gen.compile(loss='binary_crossentropy', optimizer=Adam(lr= 0.000008))

gen.summary ()

#Discriminator

#Simple forward convolutional neural network to identify fake images from real images
#Use of strides instead of maxpooling layers
#Using dropout layers to generalize 
#Use of Leaky Relu as per the suggestion of the original GAN Papers
#Learning rate lesser than that of Generator, to prevent it from getting too strong to early, and not allowing the generator to catch up

input_des = Input (shape = (32,32,1))
c1 = Convolution2D(64, 5, strides = 2,kernel_initializer='glorot_normal', activation = LeakyReLU(0.2), padding = 'same')(input_des)
d1 = Dropout (0.4)(c1)
c2 = Convolution2D(128, 5, strides = 2,kernel_initializer='glorot_normal', activation = LeakyReLU(0.2), padding = 'same')(d1)
d2 = Dropout (0.4)(c2)
c3 = Convolution2D(256, 5, strides = 1,kernel_initializer='glorot_normal', activation = LeakyReLU(0.2), padding = 'same')(d2)
d3 = Dropout (0.4)(c3)
f1 = Flatten()(d3)
de1 = Dense (256)(f1)
d4 = Dropout (0.4)(de1)
de3 = Dense (2,activation = 'softmax')(d4)

dis = Model (inputs = input_des , outputs = de3)
dis.compile(loss='categorical_crossentropy', optimizer=Adam(lr= 0.000005))

#generator with discriminator

#Gan Trick, forcing discriminator to learn fake images as real images

input_gan = Input(shape=[100])
intermed = gen(input_gan)
output_gan = dis (intermed)
gan = Model(input_gan, output_gan)
gan.compile(loss='categorical_crossentropy', optimizer=Adam(lr= 0.000006),verbose=1)
gan.summary()

#generates 32*32*1 numpy array of 100 uniform randomly sampled real numbers between 0,1

def generate_fake_images(batch_size):
    noise = np.random.uniform(0,1,size=[batch_size,100])
    fake_img = np.ndarray(shape=(batch_size,32,32,1),dtype=np.float32)
    for i in range (0,batch_size):
        fake_img[i] = gen.predict(noise[i].reshape(1,100),batch_size = 1)
    return fake_img

#0: Generates real+fake images with correct labels for discriminator to train on
#1: Generates fake images with incorrect labels for the GAN Trick

def data_generator(arg,batch_size):
    if arg == 0:
        index = random.sample(range(0,images.shape[0]),batch_size)
        real_img = images[index,:,:,:]
        X = np.concatenate((real_img,generate_fake_images(batch_size)))
        Y = np.zeros([2*batch_size,2])
        Y[:batch_size,1] = 1
        Y[batch_size:,0] = 0
        
        return X,Y
    
    if arg == 1:
        X = np.random.uniform(0,1,size=[batch_size,100])
        Y = np.zeros([batch_size,2])
        Y[:,1] = 1
        
        return X,Y

#Plots generated images 
#source: https://github.com/osh/KerasGAN/

def plot_gen(batch_size=16,dim=(4,4), figsize=(10,10) ):
    fake_img = generate_fake_images(batch_size)
    plt.figure(figsize=figsize)
    for i in range(batch_size):
        plt.subplot(dim[0],dim[1],i+1)
        plt.imshow(fake_img[i].reshape(32,32),cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#Plots generated losses 
#source: https://github.com/osh/KerasGAN/

def plot_loss(losses):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()
        

losses = {"d":[], "g":[]}

def train(nb_epoch, freq,batch_size):

    for e in tqdm(range(nb_epoch)):  
        
        #Clip discriminator weights and repeatedly train discriminator to prevent early convergence
        #Train Discriminator more than Generator
        #implementation as per Wasserstein GAN implementation
        
        dis.trainable = True
        for i in range (0,5):
            weights = [np.clip(w, -0.01, 0.01) for w in dis.get_weights()]
            dis.set_weights(weights)
            X,Y = data_generator (0,batch_size)
            dloss = dis.train_on_batch(X,Y)            
            i+=1
        
        #Generate generator loss, stop discriminator from training, pass discriminator gradient descent to generator to improve its performance
        #Trick the discriminator 
        
        X,Y = data_generator(1,batch_size)
        dis.trainable = False
        gloss = gan.train_on_batch(X,Y)
        
        if e%50 == 0 or e<50:
            dis.trainable = True
            X,Y = data_generator (0,batch_size)
            dloss = dis.train_on_batch(X,Y)
            
            X,Y = data_generator(1,batch_size)
            dis.trainable = False
            gloss = gan.train_on_batch(X,Y)
        
        losses["d"].append(dloss)
        losses["g"].append(gloss)
        
        # Update plots after frequency - 1 
        if e%freq==freq-1:
            plot_loss(losses)
            plot_gen()
            
train(2000,50,32)














