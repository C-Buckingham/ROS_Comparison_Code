#!/usr/bin/env python

import rospy
import cv2
import numpy
import threading

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

no_input = True
no_base_image = True

class video_get:

    def __init__(self):
        cv2.namedWindow("Image window", 1)
        cv2.startWindowThread()
        self.bridge = CvBridge()
        
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_color",
            Image,
            callback=self.image_callback
        )
            
    def image_callback(self, img):
        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e
            
        global no_input
        global no_base_image
        
        if no_base_image:
            print 'No image to compare against.'
            if no_input == False:
                print "Screen Grab"
                screen_grab = video_image
                cv2.imwrite("Base.jpg", screen_grab)            
                no_input = True  
                no_base_image = False
                
        else:            
            base_image = cv2.imread("/home/chris/catkin_ws/src/ROS_Comparison_Code/Base.jpg")
            hsv_base_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2HSV)             
            
            hsv_video_image = cv2.cvtColor(video_image, cv2.COLOR_BGR2HSV)
            
            if no_input == False:
                print "Screen Grab"
                screen_grab = video_image
                cv2.imwrite("Base.jpg", screen_grab)            
                no_input = True    
    
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
     
            bgr_avg_correlation = numpy.mean(bgr_comparison_result)
            hsv_avg_correlation = numpy.mean(hsv_comparison_result)
            
            print ('bgr: ', bgr_avg_correlation)
    #        print '===='
            print ('hsv: ', hsv_avg_correlation)
            
            if bgr_avg_correlation < 0.85: #+hsv_avg_correlation)/2 < 0.85:
                print 'Different'
            else:
                print 'Same' 
                   
            print '===='        
            
            cv2.imshow("Image window", video_image)        
        
    def signal_user_input():
        global no_input
        i = raw_input()
        no_input = False
    

    threading.Thread(target = signal_user_input).start()
                 
video_get()
rospy.init_node('image_get', anonymous=True)
rospy.spin()
    
cv2.destroyAllWindows()