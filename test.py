# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sklearn.metrics as metrics
import scipy.io as sio
from model import MSCNN
from VGG16 import VGG16_nodecoder
from ResNet import ResNet50_Crowd
import dataSHTB
import matplotlib.pyplot as plt
import time
import data


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted result.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)

    return mse, mae

def testResnet(num):

    #name = 'data\\timg3.jpg'
    #model = MSCNN((224, 224, 3))
    #model = VGG16_nodecoder((size, size, 3))
    model = ResNet50_Crowd((768, 1024, 3))
    model.load_weights('./SHTB_Resnet/weights-improvement-18-3.47.hdf5')

    plt.figure()

    plt.subplot(131)
    name = './SHTB/train_data/images/IMG_{}.jpg'.format(num)
    img = cv2.imread(name)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)

    plt.subplot(132)
    ground = dataSHTB.map_pixels(num, 128, 96)
    plt.imshow(ground[:, :])
    plt.tight_layout()

    plt.subplot(133)
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    dmap = model.predict(img)[0][:, :, 0]
    #dmap = cv2.GaussianBlur(dmap, (15, 15), 0)
    plt.imshow(dmap[:, :])
    plt.tight_layout()
    plt.show()

    #visualization(img[0], dmap)
    number, _ = dataSHTB.read_annotations(num)
    print('Real Count: ', number[0][0][0][0])
    print('Ground Count:', int(np.sum(ground)))
    print('NN Count:', int(np.sum(dmap)))

def testVGG():

    model = VGG16_nodecoder((224, 224, 3))
    model.load_weights('./VGG16/final_weights.h5')
    for i in range(1,5):
        name = './data/mall_dataset/frames/seq_00000{}.jpg'.format(i)
        img = cv2.imread(name)
        img = cv2.resize(img, (224, 224))
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        _start = time.time()
        dmap = model.predict(img)[0][:, :, 0]
        _stop = time.time()-_start
        print(_stop)

def read_test_annotations(num):

    data = sio.loadmat('./SHTB/test_data/ground_truth/GT_IMG_{}.mat'.format(num))
    image_info = data['image_info'][0][0]
    number = image_info['number']
    location = image_info['location']

    return number, location

def map_pixels(num, sx, sy):

    gaussian_kernel = 15

    pixels = np.zeros((sy, sx))

    number, location = read_test_annotations(num)

    for a in location[0][0]:
        #print(a)
        x, y = int(a[0] * sx / 1024), int(a[1] * sy / 768)
        if y >= sy or x >= sx:
            print("{},{} is out of range, skipping annotation for {}".format(x, y, num))
        else:
            pixels[y, x] += 100

    pixels = cv2.GaussianBlur(pixels, (gaussian_kernel, gaussian_kernel), 0)

    return pixels

def eval_ResnetCrowd():
    model = ResNet50_Crowd((768, 1024, 3))
    model.load_weights('./SHTB_Resnet/weights-improvement-18-3.47.hdf5')

    mse = 0
    mae = 0

    for num in range(1, 317):
        img_name = './SHTB/test_data/images/IMG_{}.jpg'.format(num)
        img = cv2.imread(img_name)
        density_map = map_pixels(num, 128, 96)
        img = img / 255.
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        _start = time.time()
        dmap = model.predict(img)[0][:, :, 0]
        _stop = time.time()-_start
        #images.append(density_map)
        #labels.append(dmap)
        a, b = eva_regress(density_map, dmap)
        mse = mse + a
        mae = mae + b
        print(num, _stop)

    print("mse: ", mse / 316)
    print("mae: ", mae / 316)

def CrowdResNetPic(img):
    model = ResNet50_Crowd((768, 1024, 3))
    model.load_weights('./SHTB_Resnet/weights-improvement-18-3.47.hdf5')

    plt.figure()
    #plt.clf()

    plt.subplot(121)
    #name = './SHTB/train_data/images/IMG_{}.jpg'
    #img = cv2.imread(name)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)

    plt.subplot(122)
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    dmap = model.predict(img)[0][:, :, 0]
    #dmap = cv2.GaussianBlur(dmap, (15, 15), 0)
    plt.imshow(dmap[:, :])
    plt.tight_layout()

    plt.show()

    print('NN Count:', int(np.sum(dmap)))

if __name__ == '__main__':

    #testResnet(1)
    #eval_ResnetCrowd()
    #name = './TestImage/seq_2902_2724.jpg'
    #img = cv2.imread(name)
    #img = cv2.resize(img, (1024, 768))
    #CrowdResNetPic(img)
    testVGG()
    '''
    cap = cv2.VideoCapture("QQ.mp4")
    
    while 1:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1024, 768))
        CrowdResNetPic(frame)
    '''