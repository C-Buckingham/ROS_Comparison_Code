#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
    
    def __init__(self):
        cv2.namedWindow("Image window", 1)
        cv2.startWindowThread()
        self.bridge = CvBridge()
        
        self.image_sub = rospy.Subscriber(
            "/upper_body_detector/image",
            Image,
            callback=self.image_callback
        )
        
        self.person_sub = rospy.Subscriber(
            "/upper_body_detector/detections",
            UpperBodyDetector,
            callback=self.person_callback
        )
        
    def person_callback(self, person):
#        self.person_x = person.x
#        self.person_y = person.y
        self.person_height = person.height        
        self.person_width = person.width
    
    def image_callback(self, img):
        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e
            
        print self.person_height            
        print self.person_width
        
        crop_image = video_image[0:self.person_height, 0:self.person_width]        
        
        cv2.imshow("Image window", crop_image)  

image_converter()
rospy.init_node('image_converter', anonymous=True)
rospy.spin()
cv2.destroyAllWindows()