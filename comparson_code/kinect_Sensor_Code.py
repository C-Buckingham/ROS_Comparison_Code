#!/usr/bin/env python

import rospy
import cv2
import numpy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#import Tkinter as tk
#
#def onKeyPress(event):
#    text.insert('end', 'You pressed %s\n' % (event.char, ))

base_image = cv2.imread("/home/chris/catkin_ws/src/comparson_code/Screenshot from 2015-12-28 14:25:55.png")
hsv_base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)

blue_base_image_hist = cv2.calcHist([base_image],[0],None,[256],[0,256])
green_base_image_hist = cv2.calcHist([base_image],[1],None,[256],[0,256])
red_base_image_hist = cv2.calcHist([base_image],[2],None,[256],[0,256])

hue_base_image_hist = cv2.calcHist([hsv_base_image],[0],None,[256],[0,256])
saturation_base_image_hist = cv2.calcHist([hsv_base_image],[1],None,[256],[0,256])

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
            
        hsv_video_image = cv2.cvtColor(video_image, cv2.COLOR_BGR2HSV)
            
#        print numpy.mean(cv_image[:, :, 0])
#        print numpy.mean(cv_image[:, :, 1])
#        print numpy.mean(cv_image[:, :, 2])
#        
#        print '===='
        blue_mean = numpy.mean(video_image[:, :, 0])
        green_mean = numpy.mean(video_image[:, :, 1])
        red_mean = numpy.mean(video_image[:, :, 2])
        
        bgr_mean = (blue_mean + green_mean + red_mean) / 3      
        
#        print bgr_mean
#        print '===='
        
        blue_video_hist = cv2.calcHist([video_image],[0],None,[256],[0,256])
        green_video_hist = cv2.calcHist([video_image],[1],None,[256],[0,256])
        red_video_hist = cv2.calcHist([video_image],[2],None,[256],[0,256])     
        
        blue_histogram_comparison_result = cv2.compareHist(blue_base_image_hist, blue_video_hist, cv2.cv.CV_COMP_CORREL)        
        green_histogram_comparison_result = cv2.compareHist(green_base_image_hist, green_video_hist, cv2.cv.CV_COMP_CORREL)   
        red_histogram_comparison_result = cv2.compareHist(red_base_image_hist, red_video_hist, cv2.cv.CV_COMP_CORREL)   
        
                
        hue_video_image_hist = cv2.calcHist([hsv_video_image],[0],None,[256],[0,256])
        saturation_video_image_hist = cv2.calcHist([hsv_video_image],[1],None,[256],[0,256]) 
        
        hue_histogram_comparison_result = cv2.compareHist(hue_base_image_hist, hue_video_image_hist, cv2.cv.CV_COMP_CORREL)        
        saturation_histogram_comparison_result = cv2.compareHist(saturation_base_image_hist, saturation_video_image_hist, cv2.cv.CV_COMP_CORREL)         
        
        
        bgr_avg_correlation = (blue_histogram_comparison_result+green_histogram_comparison_result+red_histogram_comparison_result)/3        
        hsv_avg_correlation = (hue_histogram_comparison_result+saturation_histogram_comparison_result)/2
        
#        print ('bgr: ', bgr_avg_correlation)
#        print '===='
#        print ('hsv: ', hsv_avg_correlation)
        
        if (bgr_avg_correlation+hsv_avg_correlation)/2 < 0.85:
            print 'Different'
        else:
            print 'Same'                
#
#        print '===='        
        
        cv2.imshow("Image window", video_image)      

video_get()
rospy.init_node('image_get', anonymous=True)
rospy.spin()
cv2.destroyAllWindows()
