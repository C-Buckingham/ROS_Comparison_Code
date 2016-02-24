#!/usr/bin/env python

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

no_base_image = True
count = 0
max_Values = [0, 0, 0]
min_Values = [1, 1, 1]

choice = raw_input('Choose between Feature Matching (F) or Colour Matching (C): ')

class person_comparison:

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

    def colour_Matching(base_image, video_image):
#        print "Colour Matching"
        hsv_base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)

        hsv_video_image = cv2.cvtColor(video_image, cv2.COLOR_BGR2HSV)

        global max_Values
        global min_Values

        bgr_comparison_result = []
        hsv_comparison_result = []
        for x in range (0, 3):
            bgr_video_image_hist = cv2.calcHist([video_image], [x], None, [256], [0,256])
            bgr_base_image_hist = cv2.calcHist([base_image], [x], None, [256], [0,256])
            bgr_comparison_result.append(cv2.compareHist(bgr_video_image_hist, bgr_base_image_hist, cv2.cv.CV_COMP_CORREL))
            #print bgr_comparison_result[x]
            if (bgr_comparison_result[x] < 1):
                max_Values[x] = max(max_Values[x], bgr_comparison_result[x])
                
            min_Values[x] = min(min_Values[x], bgr_comparison_result[x])
            if (x < 2):
                hsv_video_image_hist = cv2.calcHist([hsv_video_image],[x],None,[256],[0,256])
                hsv_base_image_hist = cv2.calcHist([hsv_base_image],[x],None,[256],[0,256])
                hsv_comparison_result.append(cv2.compareHist(hsv_video_image_hist, hsv_base_image_hist, cv2.cv.CV_COMP_CORREL))

        bgr_avg_correlation = np.mean(bgr_comparison_result)
        hsv_avg_correlation = np.mean(hsv_comparison_result)

        with open("Output.txt", "w") as text_file:
            text_file.write("Min BGR Value: %s"% min_Values)
            text_file.write("\nMax BGR Value: %s"% max_Values)

        print ('bgr: ', bgr_avg_correlation)
#        print '===='
        print ('hsv: ', hsv_avg_correlation)

        if bgr_avg_correlation > 0.85 or hsv_avg_correlation > 0.85:
            print 'Same'
        else:
            print 'Different'

        print '==='

    def feature_Matching():
        print "Feature Matching"

        kp1, des1 = orb.detectAndCompute(video_image,None)
        kp2, des2 = orb.detectAndCompute(base_image, None)

#                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher()

        match = bf.match(des1,des2)
        matches = bf.knnMatch(des1, des2)

#                    Why I am using ORB: https://www.willowgarage.com/sites/default/files/orb_final.pdf

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        match = sorted(match, key = lambda x:x.distance)

        video_image = cv2.drawKeypoints(video_image,kp1,color=(0,255,0), flags=0)
        base_image = cv2.drawKeypoints(base_image,kp2,color=(0,255,0), flags=0)

    global options

    options = {'C':colour_Matching, 'F':feature_Matching}

    def image_callback(self, img, person):

        person_pos_x = person.pos_x
        person_pos_y = person.pos_y
        person_height = person.height
        person_width = person.width
        
        def clamp(n, minn, maxn):
            return max(min(maxn, n), minn)

        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e

        global screen_grab
        global no_base_image
        global count
        global choice

        if (person_height):

            person_h = min(person_height[0], 480)
            person_w = min(person_width[0], 640)
            person_x = min(person_pos_x[0], 480)
            person_y = min(person_pos_y[0], 640)

            person_h = max(person_height[0], 0)
            person_w = max(person_width[0], 0)
            person_x = max(person_pos_x[0], 0)
            person_y = max(person_pos_y[0], 0)

            count = count + 1
            print count

        if (count == 20):
            screen_grab = video_image[person_y:(person_y+person_w)*2, person_x:person_x+person_h]
            no_base_image = False
#            count = 0

        if (no_base_image == False):

            base_image = screen_grab
            if (person_pos_y):
                video_image = video_image[person_y:(person_y+person_w)*2, person_x:person_x+person_h]
                options[choice](base_image, video_image)

            cv2.imshow("Live Image", video_image)
            cv2.imshow("Base Image", base_image)

person_comparison()
rospy.init_node('person_comparison', anonymous=True)
rospy.spin()

cv2.destroyAllWindows()
