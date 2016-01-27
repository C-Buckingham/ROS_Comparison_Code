#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError

class image_crop:
    
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
        self.person_x = person.pos_x
        self.person_y = person.pos_y
        self.person_height = person.height        
        self.person_width = person.width
    
    def image_callback(self, img):
        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e
        
        if (self.person_height):
            
            if (self.person_height[0] < 0):
                p_h = 0
            elif(self.person_height[0] > 480):
                p_h = 480
            else:
                p_h = self.person_height[0]
                
            if (self.person_width[0] < 0):
                p_w = 0
            elif(self.person_width[0] > 640):
                p_w = 640
            else:
                p_w = self.person_width[0]
   
            if (self.person_x[0] < 0):
                p_x = 0
            elif(self.person_x[0] > 480):
                p_x = 480
            else:
                p_x = self.person_x[0]
                
            if (self.person_y[0] < 0):
                p_y = 0
            elif(self.person_y[0] > 640):
                p_x = 640
            else:
                p_y = self.person_y[0]
            
            print p_h            
            print p_w
            print p_x            
            print p_y
            
            crop_image = video_image[p_y:(p_y+p_w)*2, p_x:p_x+p_h]        
        
            cv2.imshow("Image window", crop_image)  
        
        print "No Person"

image_crop()
rospy.init_node('image_converter', anonymous=True)
rospy.spin()
cv2.destroyAllWindows()