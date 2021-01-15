# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author:XLin
# datetime:2020/3/27 0027 23:01
# software: PyCharm
import cv2
import numpy as np

img = cv2.imread("3-.png", 0)
cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))
cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()
