#!/usr/bin/env python

import rospy
import cv2
import numpy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#
class image_get:

    def __init__(self):
        cv2.namedWindow("Image window", 1)
        cv2.startWindowThread()
        self.bridge = CvBridge()
        
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_color",
            Image,
            callback=self.iamge_callback
        )
    
    def iamge_callback(self, img):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e
            
        cv2.imshow("Image window", cv_image)

image_get()
rospy.init_node('image_get', anonymous=True)
rospy.spin()
cv2.destroyAllWindows()
