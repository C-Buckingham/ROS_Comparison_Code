#!/usr/bin/env python

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

i = 0

class match_output:

    def __init__(self):
        self.bridge = CvBridge()

        match_sub = Subscriber(
            "/person_comparison/match_image_out",
            Image,
            queue_size=1
        )

        base_sub = Subscriber(
            "/person_comparison/base_image_out",
            Image,
            queue_size=1
        )

        ts = ApproximateTimeSynchronizer([match_sub, base_sub], 1, 0.1)
        ts.registerCallback(self.image_callback)

    def image_callback(self, match_img, base_img):
        global i
        try:
            match_image = self.bridge.imgmsg_to_cv2(match_img, "bgr8")
            base_image = self.bridge.imgmsg_to_cv2(base_img, "bgr8")

        except CvBridgeError, e:
            print e

        if i == 0:
            cv2.imwrite("/home/chris/catkin_ws/src/ROS_Comparison_Code/Result_Images/Initial_Test/Base_Image.jpg", base_image)

        cv2.imwrite('/home/chris/catkin_ws/src/ROS_Comparison_Code/Result_Images/Initial_Test/Match{:>05}.jpg'.format(i), match_image)
        i += 1

rospy.init_node('match_output')
match_output()
rospy.spin()
cv2.destroyAllWindows()
