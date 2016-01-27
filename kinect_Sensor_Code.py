#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import threading
import Image

from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
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
        
        self.person_sub = rospy.Subscriber(
            "/upper_body_detector/detections",
            UpperBodyDetector,
            callback=self.person_callback
        )
        
    def person_callback(self, person):
        self.person_x = person.pos_x
        self.person_y = person.pos_y
        self.person_height = person.height        
        self.person_width = person.width
            
    def image_callback(self, img):
        try:
            video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e
            
        global no_input
        global no_base_image

        crop = False                
                
        if (self.person_height):
            if (self.person_height[0] < 0):
                p_h = 0
            elif(self.person_height[0] > 480):
                p_h = 480
            else:
                p_h = self.person_height[0]
                
            if (self.person_width[0] < 0):
                p_w = 0
            elif(self.person_width[0] > 640):
                p_w = 640
            else:
                p_w = self.person_width[0]
   
            if (self.person_x[0] < 0):
                p_x = 0
            elif(self.person_x[0] > 480):
                p_x = 480
            else:
                p_x = self.person_x[0]
                
            if (self.person_y[0] < 0):
                p_y = 0
            elif(self.person_y[0] > 640):
                p_y = 640
            else:
                p_y = self.person_y[0]
        
        if no_base_image:
            print 'No image to compare against.'
            if no_input == False:
                print "Screen Grab"
                
                if (self.person_height):
                    screen_grab = video_image[p_y:(p_y+p_w)*2, p_x:p_x+p_h]
                else:
                    screen_grab = video_image
                    
                cv2.imwrite("Base.jpg", screen_grab)            
                no_input = True  
                no_base_image = False
                
        else:
            
            if(self.person_height):
                crop = True
                new_size = (640, 480)
                old_size = video_image.size
                new_image = Image.new("RGB", new_size)
                video_image = video_image[p_y:(p_y+p_w)*2, p_x:p_x+p_h]
                new_image.paste(video_image, (new_size[0]-old_size[0]/2,
                                           new_size[1]-old_size[1]/2))
#                http://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python
             
             
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
                
            bgr_avg_correlation = np.mean(bgr_comparison_result)
#            hsv_avg_correlation = np.mean(hsv_comparison_result)
            
#            print ('bgr: ', bgr_avg_correlation)
    #        print '===='
#            print ('hsv: ', hsv_avg_correlation)
            
            if bgr_avg_correlation < 0.85: #+hsv_avg_correlation)/2 < 0.85:
                print 'Different'
            else:
                print 'Same' 
                   
            print '===='        
            if (crop):
                vis = np.concatenate((base_image, new_image), axis=1)
            else:
                vis = np.concatenate((base_image, video_image), axis=1)
                            
            cv2.imshow("Image window", vis)        
        
    def signal_user_input():
        global no_input
        i = raw_input()
        no_input = False
    

    threading.Thread(target = signal_user_input).start()
                 
video_get()
rospy.init_node('image_get', anonymous=True)
rospy.spin()
    
cv2.destroyAllWindows()