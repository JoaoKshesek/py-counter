import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

img = cv2.imread('C:/Users/Thiago/Desktop/PYTHON/puppies.jpg')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(img1)

box, label, count = cv.detect_common_objects(img)
output = draw_bbox(img, box, label, count)

output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(output)
plt.show()

print("Number of objects in this image are " + str(len(label))) 