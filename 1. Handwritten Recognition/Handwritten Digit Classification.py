import cv2
import numpy as np
from preprocessors import x_cord_contour, makeSquare, resize_to_pixel
from keras.models import load_model

classifier = load_model(r'E:\Python\[FreeCourseSite.com] Udemy - Deep Learning Computer Vision™ CNN, OpenCV, YOLO, SSD & GANs\DeepLearningCV\Trained Models\mnist_simple_cnn.h5')

image = cv2.imread(r'E:\Computer Vision\1. Handwritten Recognition\numbers.jpg')
#image  = cv2.resize(image, (250,150))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow("Image",image)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray, (5,5), 0)

edged = cv2.Canny(blurred, 30 , 150)
#cv2.imshow("edged",edged)
#cv2.waitKey(0)

(contours, _)   = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key = x_cord_contour, reverse = False)

full_number = []

for c in contours:
    (x,y,w,h) = cv2.boundingRect(c)

    if w>=5 and h>=25 :
        roi = blurred[y:y+h, x: x+w]
        ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        roi = makeSquare(roi)
        roi = resize_to_pixel(28,roi)
        cv2.imshow("ROI",roi)
        roi = roi/255.0
        roi = roi.reshape(1,28,28,1)

        res = str(classifier.predict_classes(roi,1, verbose=0)[0])
        full_number.append(res)
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(image, res,  (x,y+155), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)
        cv2.imshow("Image",image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
print("The number is : " + "".join(full_number))
