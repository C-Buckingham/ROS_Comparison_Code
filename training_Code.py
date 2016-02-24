#!/usr/bin/env python

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

class training:

    def __init__(self):
        cv2.namedWindow("Live Image", 1)
        cv2.namedWindow("Base Image", 1)
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

        ts = ApproximateTimeSynchronizer([image_sub, person_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

    def image_callback(self, img, person):

        person_pos_x = person.pos_x
        person_pos_y = person.pos_y
        person_height = person.height
        person_width = person.width

        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e

        if (person_height):

            person_h = min(person_height[0], 480)
            person_w = min(person_width[0], 640)
            person_x = min(person_pos_x[0], 480)
            person_y = min(person_pos_y[0], 640)

            person_h = max(person_height[0], 0)
            person_w = max(person_width[0], 0)
            person_x = max(person_pos_x[0], 0)
            person_y = max(person_pos_y[0], 0)

        video_image = video_image[person_y:(person_y+person_w)*2, person_x:person_x+person_h]

        

training()
rospy.init_node('training', anonymous=True)
rospy.spin()

cv2.destroyAllWindows()
