import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from keras.layers import UpSampling2D, Dropout, Activation, Flatten
from keras import optimizers
import numpy as np
from keras.models import Model

def VGG16_nodecoder(input_shape):
    # input_shape = (768, 1024, 3)
    inputs = Input(input_shape)

    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)

    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(1, 3, activation='relu', padding='same')(x)

    x = UpSampling2D(size=(8, 8), interpolation='nearest')(x)

    #x = Conv2DTranspose(64, 2, strides=2, padding='valid',
                        #output_padding=0, use_bias=True)(x)
    #x = Conv2DTranspose(32, 2, strides=2, padding='valid',
                        #output_padding=0, use_bias=True)(x)
    #x = Conv2DTranspose(16, 2, strides=2, padding='valid',
                        #output_padding=0, use_bias=True)(x)

    #x = Conv2D(1, 3, activation='relu', padding='same')(x)


    model = Model(inputs=inputs, outputs=x)

    return model

if __name__ == '__main__':
    model = VGG16_nodecoder((768, 1024, 3))
    model.summary()
