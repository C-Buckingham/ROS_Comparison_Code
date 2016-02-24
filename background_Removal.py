#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import math

from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

class training:

    def __init__(self):
        cv2.namedWindow("Live Image", 1)
        cv2.namedWindow("Depth Image", 1)
        cv2.startWindowThread()
        self.bridge = CvBridge()

        image_sub = Subscriber(
            "/camera/rgb/image_color",
            Image,
        )

        person_sub = Subscriber(
            "/upper_body_detector/detections",
            UpperBodyDetector,
        )
        
        depth_sub = Subscriber(
            "/camera/depth/image_rect",
            Image,
        )

        ts = ApproximateTimeSynchronizer([image_sub, person_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

    def image_callback(self, img, person, depth):

        person_pos_x = person.pos_x
        person_pos_y = person.pos_y
        person_height = person.height
        person_width = person.width
        person_depth = person.median_depth    
        depth_image = []
        
        if (person_height):

            person_h = min(person_height[0], 480)
            person_w = min(person_width[0], 640)
            person_x = min(person_pos_x[0], 480)
            person_y = min(person_pos_y[0], 640)

            person_h = max(person_height[0], 0)
            person_w = max(person_width[0], 0)
            person_x = max(person_pos_x[0], 0)
            person_y = max(person_pos_y[0], 0)
                    
        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth)
        except CvBridgeError, e:
            print e
            
        if(person_pos_y):
            video_image = video_image[person_y:(person_y+person_w)*2, person_x:person_x+person_h]
            depth_image = depth_image[person_y:(person_y+person_w)*2, person_x:person_x+person_h]

#        print depth_image[0][1]
#        print "Width: ", depth_image.shape[0]
#        print "Height: ", depth_image.shape[1]
#        (thresh, depth_image) = cv2.threshold(depth_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#        depth_image = cv2.bitwise_and(video_image,video_image,mask = depth_image)
        
        if(person_pos_y):        
            for x in range (0,depth_image.shape[0]):
                for y in range (0, depth_image.shape[1]):
                    if(depth_image[x][y] < person_depth[0]+1):          
                        video_image[x][y] = video_image[x][y]
                    else:
#                        print depth_image[x][y]
                        video_image[x][y] = 0
#                    
        
#        print depth_image
#        print video_image
#        print depth_image[0]
#        print type(depth_image)
        
        
        if(person_pos_y):        

#            depth_image = cv2.multiply(depth_image, video_image[0])
            cv2.imshow("Live Image", video_image)
            cv2.imshow("Depth Image", depth_image)

training()
rospy.init_node('training', anonymous=True)
rospy.spin()

cv2.destroyAllWindows()
