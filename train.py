# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(122)

import os
import sys
import argparse
import pandas as pd
import time

from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split

from model import MSCNN
from VGG16 import VGG16_nodecoder
from ResNet import ResNet50_Crowd
from dataSHTB import generator

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=1,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=100,
        help="The number of train iterations.")

    args = parser.parse_args()

    train(int(args.batch), int(args.epochs),int(args.size))


def train(batch, epochs, size):
    """Train the model.

    Arguments:
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        size: Integer, image size.
    """
    if not os.path.exists('model'):
        os.makedirs('model')

    #model = MSCNN((size, size, 3))
    #model = VGG16_nodecoder((size, size, 3))
    model = ResNet50_Crowd((768, 1024, 3))

    opt = SGD(lr=1e-5, momentum=0.9, decay=0.0005)
    model.compile(optimizer=opt, loss='mse',metrics=['acc','mse','mae'])
    lr = ReduceLROnPlateau(monitor='loss', min_lr=1e-7)

    indices = list(range(1, 401))
    train, test = train_test_split(indices, test_size=0.25)

    model_name = "chaptcha-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

    filepath = "./SHTB_Resnet/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')

    hist = model.fit_generator(
        generator(train, batch, size),
        validation_data=generator(test, batch, size),
        steps_per_epoch=len(train) // batch,
        validation_steps=len(test) // batch,
        epochs=epochs,
        callbacks=[lr,
                   tensorboard,
                   checkpoint],
        shuffle=True)

    model.save('model\\final_weights.h5')

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model\\history.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv)
