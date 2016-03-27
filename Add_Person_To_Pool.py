#!/usr/bin/env python

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

bgr_hist = []

class training:
    def __init__(self):
        cv2.namedWindow("Live Image", 1)
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
            queue_size=1
        )

        ts = ApproximateTimeSynchronizer([image_sub, person_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.image_callback)

    def image_callback(self, img, person, depth):
        global bgr_hist

        person_pos_x = person.pos_x
        person_pos_y = person.pos_y
        person_height = person.height
        person_width = person.width

        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth)
        except CvBridgeError, e:
            print e

        if len(person_height) > 0:

            person_h = min(person_height[0], 480)
            person_w = min(person_width[0], 640)
            person_x = min(person_pos_x[0], 480)
            person_y = min(person_pos_y[0], 640)

            person_h = max(person_height[0], 0)
            person_w = max(person_width[0], 0)
            person_x = max(person_pos_x[0], 0)
            person_y = max(person_pos_y[0], 0)

        depth_offset_percentage = float(person_x / 640.0 * 100.0)

        depth_image = depth_image.transpose()
        depth_image = np.roll(depth_image, 35)
        depth_image = depth_image.transpose()

        if depth_offset_percentage >= 70:
            depth_image = np.roll(depth_image, -15)
        elif depth_offset_percentage <= 30:
            depth_image = np.roll(depth_image, 15)

        video_image = video_image[person_y:(person_y+person_w)*2, person_x:person_x+person_h]

        depth_image = depth_image[person_y:(person_y + person_w) * 2, person_x:person_x + person_h]

        depth_image_shape = depth_image.shape
        depth_image = depth_image.flatten()
        depth_image = np.where(depth_image < person.median_depth[0] * 1.10, 1, 0)
        depth_image = np.reshape(depth_image, depth_image_shape)

        for x in range(0, 3):
            video_image[:, :, x] = video_image[:, :, x] * depth_image

        cv2.imshow("Live Image", video_image)

        for x in range(0, 3):
            bgr_hist.append(cv2.calcHist([video_image], [x], None, [256], [1, 256]))

training()
rospy.init_node('training', anonymous=True)
rospy.spin()

cv2.destroyAllWindows()

with open("Histogram Pool.txt", "a") as text_file:
    text_file.write("=\n")

for x in range(0, 3):
    with open("Histogram Pool.txt", "a") as text_file:
        text_file.write("?")
        text_file.write("\n%s\n" % bgr_hist[x])