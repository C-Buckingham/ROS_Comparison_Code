#!/usr/bin/env python

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from multiprocessing import Pool

class Machine_Learner:

    def __init__(self):
        # Create new windows
        cv2.namedWindow("Live Image", 1)
        cv2.namedWindow("Base Image", 1)
        #cv2.namedWindow("Original Image", 1)
        cv2.startWindowThread()
        self.bridge = CvBridge()

        image_sub = Subscriber(
            "/camera/rgb/image_color",
            Image,
            queue_size=1
        )

        person_sub = Subscriber(
            "/upper_body_detector/detections",
            UpperBodyDetector,
            queue_size=1
        )

        depth_sub = Subscriber(
            "/camera/depth/image_rect",
            Image,
            queue_size=1
        )

        # Time syncronizer is implimented to make sure that all of the frames match up from all of the topics.
        ts = ApproximateTimeSynchronizer([image_sub, person_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.image_callback)

    def image_callback(self, img, person, depth):
        # Convert the messages from the sensor into an image
        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth)
        except CvBridgeError, e:
            print e

        x = 0

        cv2.imshow("Live Image", video_image)

        # Making sure that the box stays within the bounds of the image
        person_h = min(person.height[x], 480)
        person_w = min(person.width[x], 640)
        person_x = min(person.pos_x[x], 480)
        person_y = min(person.pos_y[x], 640)

        person_h = max(person.height[x], 0)
        person_w = max(person.width[x], 0)
        person_x = max(person.pos_x[x], 0)
        person_y = max(person.pos_y[x], 0)

        thread_Pool = Pool(len(person.height))

        # Calculate the rough section of the screen the person is
        depth_offset_percentage = float(person_x / 640.0 * 100.0)

        # Transpose the array to move the mask correctly
        depth_image = depth_image.transpose()

        # Roll the mask to try and offset the error from the sensors
        depth_image = np.roll(depth_image, 35)

        # Put the array back into the correct orientation
        depth_image = depth_image.transpose()

        if depth_offset_percentage >= 70:
            depth_image = np.roll(depth_image, -15)
        elif depth_offset_percentage <= 30:
            depth_image = np.roll(depth_image, 15)

        # Crop the depth image
        depth_image = depth_image[person_y:(person_y + person_w) * 2, person_x:person_x + person_h]

        # Store the size of the depth image
        depth_image_shape = depth_image.shape

        # Flatten the array into a 1D shape
        depth_image = depth_image.flatten()

        # Replace all the values in the array with either 1 or 0 for masking to work properly
        depth_image = np.where(depth_image < person.median_depth[x] * 1.10, 1, 0)

        # Reshape the array into to correct shape
        depth_image = np.reshape(depth_image, depth_image_shape)

        # Crop the base image
        base_image = video_image[person_y:(person_y + person_w) * 2, person_x:person_x + person_h]

        # Loop to go through and apply the mask to the base image
        for y in range(0, 3):
            base_image[:, :, y] = base_image[:, :, y] * depth_image

        base_image = np.ma.masked_where(base_image == 0, base_image)

        cv2.imshow("Base Image", base_image)
        print "Blue Mean:  ", np.mean(base_image[:, :, 0])
        print "Green Mean: ", np.mean(base_image[:, :, 1])
        print "Red Mean:   ", np.mean(base_image[:, :, 2])

rospy.init_node('Machine_Learner')
Machine_Learner()
rospy.spin()
cv2.destroyAllWindows()