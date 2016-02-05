#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import threading

from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

no_input = True
no_base_image = True
count = 0

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

    def image_callback(self, img, person):
        
        person_x = person.pos_x
        person_y = person.pos_y
        person_height = person.height        
        person_width = person.width
        
        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e
            
        global no_input
        global screen_grab
        global no_base_image
        global count
        global choice
        
        if (person_height):
            max(person_height, 480)
            max(person_width, 640)
            max(person_x, 480)
            max(person_y, 640)
            
            min(person_height, 0)
            min(person_width, 0)
            min(person_x, 0)
            min(person_y, 0)
                            
        print "Press Enter to capture base image"
        
        if no_input == False:
            print "No Person"
            if (person_height):
                count = count + 1
                print count                    
                
            if (count > 20):
                screen_grab = video_image[person_y[0]:(person_y[0]+person_width[0])*2, person_x[0]:person_x[0]+person_height[0]]          
                no_input = True  
                no_base_image = False
            
        elif (no_base_image and no_input):
            base_image = screen_grab
            if (person_y):
                video_image = video_image[person_y[0]:(person_y[0]+person_width[0])*2, person_x[0]:person_x[0]+person_height[0]]
                if (choice == 'F'):
                    orb = cv2.ORB()
    
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
                    
                else:
                    hsv_base_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2HSV)             
                    
                    hsv_video_image = cv2.cvtColor(video_image, cv2.COLOR_BGR2HSV)
                            
                    bgr_comparison_result = []
                    hsv_comparison_result = []
                    
                    for x in range (0, 3):
                        bgr_video_image_hist = cv2.calcHist([video_image], [x], None, [256], [0,256])
                        bgr_base_image_hist = cv2.calcHist([base_image], [x], None, [256], [0,256])
                        bgr_comparison_result.append(cv2.compareHist(bgr_video_image_hist, bgr_base_image_hist, cv2.cv.CV_COMP_CORREL))
                        if (x < 2):
                            hsv_video_image_hist = cv2.calcHist([hsv_video_image],[x],None,[256],[0,256])
                            hsv_base_image_hist = cv2.calcHist([hsv_base_image],[x],None,[256],[0,256])                
                            hsv_comparison_result.append(cv2.compareHist(hsv_video_image_hist, hsv_base_image_hist, cv2.cv.CV_COMP_CORREL))
                        
                    bgr_avg_correlation = np.mean(bgr_comparison_result)
                    hsv_avg_correlation = np.mean(hsv_comparison_result)
                    
                    print ('bgr: ', bgr_avg_correlation)
            #        print '===='
                    print ('hsv: ', hsv_avg_correlation)
                    
                    if bgr_avg_correlation > 0.85 or hsv_avg_correlation > 0.85:
                        print 'Same'
                    else:
                        print 'Different' 
                           
                    print '==='
            
            cv2.imshow("Live Image", video_image)
            cv2.imshow("Base Image", base_image)
        
    def signal_user_input():
        global no_input
        i = raw_input()
        no_input = False
    

    threading.Thread(target = signal_user_input).start()

person_comparison()
rospy.init_node('person_comparison', anonymous=True)
rospy.spin()
    
cv2.destroyAllWindows()