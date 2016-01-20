#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#img = cv2.imread('/home/chris/Downloads/Gaming-PC.jpg', 0)
#
#orb = cv2.ORB()
#
#kp = orb.detect(img,None)
#
#kp, des = orb.compute(img, kp)
#
#img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
#plt.imshow(img2),plt.show()

class image_converter:
    
    def __init__(self):
        cv2.namedWindow("Image window", 1)
        cv2.startWindowThread()
        self.bridge = CvBridge()
        
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_color",
            Image,
            callback=self.image_callback
        )
            
    def image_callback(self, img):
        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e
            
        orb = cv2.ORB()

        kp = orb.detect(video_image,None)

        kp, des = orb.compute(video_image, kp)

        video_image = cv2.drawKeypoints(video_image,kp,color=(0,255,0), flags=0)
        
        cv2.imshow("Image window", video_image)  
        
image_converter()
rospy.init_node('image_converter', anonymous=True)
rospy.spin()
cv2.destroyAllWindows()