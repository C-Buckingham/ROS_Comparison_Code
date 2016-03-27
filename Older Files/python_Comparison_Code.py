#!/usr/bin/env python
"""
Spyder Editor

This temporary script file is located here:
/home/chris/.spyder2/.temp.py
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('/home/chris/catkin_ws/src/comparson_code/71P67LGRY_large.jpg')
plt.hist(image.ravel(),256,[0,256]); 
plt.show()