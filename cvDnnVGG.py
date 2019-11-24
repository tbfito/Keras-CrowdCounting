import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.figure()

plt.subplot(121)
img = cv2.imread("./data_dnnc/seq_3100_2864.jpg")
b, g, r = cv2.split(img)
img2 = cv2.merge([r, g, b])
plt.imshow(img2)

plt.subplot(122)
cvNet = cv2.dnn.readNetFromTensorflow('./method/frozen_graph_v2.pb')
cvNet.setInput(cv2.dnn.blobFromImage(img, 1. / 255, size=(224, 224)))
cvOut = cvNet.forward()
plt.imshow(cvOut[0][0])
print(cvOut.sum())
#plt.tight_layout()

plt.show()
