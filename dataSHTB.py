# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy.io as sio
from multiprocessing.dummy import Pool as ThreadPool

import matplotlib.pyplot as plt

def visualization(num):
    plt.figure()

    plt.subplot(121)
    img_name = './SHTB/train_data/images/IMG_{}.jpg'.format(num)
    img = cv2.imread(img_name)

    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    plt.imshow(img)

    plt.subplot(122)
    dmap = map_pixels(num, 128, 96)
    plt.imshow(dmap[:, :])

    plt.tight_layout()
    plt.show()

    number, _ = read_annotations(num)
    print('count: ',number)

def read_annotations(num):

    data = sio.loadmat('./SHTB/train_data/ground_truth/GT_IMG_{}.mat'.format(num))
    image_info = data['image_info'][0][0]
    number = image_info['number']
    location = image_info['location']

    return number, location

def map_pixels(num, sx, sy):

    gaussian_kernel = 15

    pixels = np.zeros((sy, sx))

    number, location = read_annotations(num)

    for a in location[0][0]:
        #print(a)
        x, y = int(a[0] * sx / 1024), int(a[1] * sy / 768)
        if y >= sy or x >= sx:
            print("{},{} is out of range, skipping annotation for {}".format(x, y, num))
        else:
            pixels[y, x] += 100

    pixels = cv2.GaussianBlur(pixels, (gaussian_kernel, gaussian_kernel), 0)

    return pixels

def get_data(num):

    img_name = './SHTB/train_data/images/IMG_{}.jpg'.format(num)
    img = cv2.imread(img_name)

    density_map = map_pixels(num, 128, 96)

    img = img / 255.

    density_map = np.expand_dims(density_map, axis=-1)

    return img, density_map

def generator(indices, batch, size):

    i = 0
    n = len(indices)

    if batch > n:
        raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch, n))

    while True:
        if i + batch >= n:
            np.random.shuffle(indices)
            i = 0
            continue

        pool = ThreadPool(2)
        res = pool.map(lambda x: get_data(x), indices[i: i + batch])
        pool.close()
        pool.join()

        i += batch
        images = []
        labels = []

        for r in res:
            images.append(r[0])
            labels.append(r[1])

        images = np.array(images)
        labels = np.array(labels)

        yield images, labels

if __name__ == '__main__':
    #pix = map_pixels(1, 1024, 768)
    #print(pix)
    #plt.imshow(pix)
    #plt.tight_layout()
    #plt.show()
    visualization(1)
