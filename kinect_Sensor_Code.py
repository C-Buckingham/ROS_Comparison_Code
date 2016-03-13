#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

tmp_array = np.array([0, 0, 0])
line_count = 0

with open('Output.txt', 'r') as text_file:
    data=text_file.read().replace('\n', '')
    data = data.replace('.', '')
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.split("===")
    data = ''.join(data)
    data = data.split("+++")
    data = np.asarray(data)
    data = filter(None, data)
    ar = data[0]
    ar = ar.split(' ')
    ar = np.asarray(ar)
    ar = filter(None, ar)
    ar = np.asarray(ar)
    
count = 0
combined_hist_values = [0, 0, 0]
base_image = []

choice = raw_input('Choose between Feature Matching (F) or Colour Matching (C): ')

class person_comparison:

    def __init__(self):
        ts = None
        cv2.namedWindow("Live Image", 1)
        cv2.namedWindow("Base Image", 1)
        cv2.namedWindow("Original Image", 1)
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
        
#        detections_image_sub = Subscriber(
#            "/upper_body_detector/image",
#            Image,
#            queue_size=1
#        )

        ts = ApproximateTimeSynchronizer([image_sub, person_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.image_callback)

    def colour_Matching(base_image, video_image, depth_image):     
               
        for y in range (0, 3):
            video_image[:, :, y] = video_image[:, :, y]*depth_image
                        
        hsv_base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)

        hsv_video_image = cv2.cvtColor(video_image, cv2.COLOR_BGR2HSV)

        global max_Values
        global min_Values

        bgr_comparison_result = []
        hsv_comparison_result = []
        for x in range (0, 3):
            bgr_video_image_hist = cv2.calcHist([video_image], [x], None, [256], [1,256])
            print type(bgr_video_image_hist)
            bgr_base_image_hist = cv2.calcHist([base_image], [x], None, [256], [1,256])
            combined_hist_values[x] = bgr_video_image_hist 
            bgr_comparison_result.append(cv2.compareHist(bgr_video_image_hist, bgr_base_image_hist, cv2.cv.CV_COMP_CORREL))         
            
            if (x < 2):
                hsv_video_image_hist = cv2.calcHist([hsv_video_image],[x],None,[256],[1,256])
                hsv_base_image_hist = cv2.calcHist([hsv_base_image],[x],None,[256],[1,256])
                hsv_comparison_result.append(cv2.compareHist(hsv_video_image_hist, hsv_base_image_hist, cv2.cv.CV_COMP_CORREL))

        bgr_avg_correlation = np.mean(bgr_comparison_result)
        hsv_avg_correlation = np.mean(hsv_comparison_result)

#        with open("Output.txt", "a") as text_file:
#            text_file.write("\nMin BGR Value: %s"% min_Values)
#            text_file.write("\nMax BGR Value: %s"% max_Values)
#            text_file.write("\nHistogram: %s"% combined_hist_values)

#        print ('bgr: ', bgr_avg_correlation)
#        print '===='
#        print ('hsv: ', hsv_avg_correlation)

        if bgr_avg_correlation > 0.85:# or hsv_avg_correlation > 0.85:
            cv2.imshow("Live Image", video_image)
            cv2.imshow("Base Image", base_image)
            print 'Same'
        else:
            print 'Different'

        print '==='
#        print "Section 9"

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

    def image_callback(self, img, person, depth):
        global count
        global choice        
        global base_image
        
        combined_hist_values = [0, 0, 0]
        
#        try:
#            original_image = self.bridge.imgmsg_to_cv2(detect_img, "bgr8")
#        except CvBridgeError, e:
#                print e      
#        
#        cv2.imshow("Original Image", original_image)
        if len(person.height) > 0:
            for x in range (0, len(person.height)):     
                person_h = None
                person_w = None
                person_x = None
                person_y = None
                depth_image = None
                video_image = None
                depth_offset_percentage = None
                
    #            print count
                
                try:
                    video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
                    depth_image = self.bridge.imgmsg_to_cv2(depth)
                except CvBridgeError, e:
                    print e      
    
                cv2.imshow("Original Image", video_image)
    
                person_h = min(person.height[x], 480)
                person_w = min(person.width[x], 640)
                person_x = min(person.pos_x[x], 480)
                person_y = min(person.pos_y[x], 640)
    
                person_h = max(person.height[x], 0)
                person_w = max(person.width[x], 0)
                person_x = max(person.pos_x[x], 0)
                person_y = max(person.pos_y[x], 0)
    
                count = count + 1
                
                depth_offset_percentage = float(person_x/640.0*100.0)
    #            print depth_offset_percentage
                
                depth_image = depth_image.transpose()
                depth_image = np.roll(depth_image, 35)
                depth_image = depth_image.transpose()
                
                if depth_offset_percentage >= 70:
                    depth_image = np.roll(depth_image, -15)
                elif depth_offset_percentage <= 30:
                    depth_image = np.roll(depth_image, 15)
                    
                
                depth_image = depth_image[person_y:(person_y+person_w)*2, person_x:person_x+person_h]
                
                depth_image_shape = depth_image.shape
                depth_image = depth_image.flatten()
                depth_image = np.where(depth_image < person.median_depth[x]*1.10, 1, 0)
    #            depth_image = np.where(depth_image > person.median_depth[x]*1.50, 1, 0)
                depth_image = np.reshape(depth_image, (depth_image_shape))                     
                
                if (count == 20 and person_h):
                    base_image = video_image[person_y:(person_y+person_w)*2, person_x:person_x+person_h]
                    for y in range (0, 3):
                        base_image[:, :, y] = base_image[:, :, y]*depth_image
                            
                elif(count == 20 and person_h == False):
                    count = 0
                elif(count > 20):
                    video_image = video_image[person_y:(person_y+person_w)*2, person_x:person_x+person_h]
                    options[choice](base_image, video_image, depth_image)
person_comparison()
rospy.init_node('person_comparison', anonymous=True)
rospy.spin()
cv2.destroyAllWindows()
with open("Output.txt", "a") as text_file:
    text_file.write("=\n")
    
for x in range(0, 3):
    with open("Output.txt", "a") as text_file:
        text_file.write("+")        
        text_file.write("\n%s"%combined_hist_values[x])
        text_file.write("\n")