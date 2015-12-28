#!/usr/bin/env python

import rospy
import cv2
import numpy
import Tkinter as tk

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import Tkinter as tk

def onKeyPress(event):
    text.insert('end', 'You pressed %s\n' % (event.char, ))

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
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError, e:
            print e
            
        print numpy.mean(cv_image[:, :, 0])
        print numpy.mean(cv_image[:, :, 1])
        print numpy.mean(cv_image[:, :, 2])
        
        print '===='
        blue_mean = numpy.mean(cv_image[:, :, 0])
        green_mean = numpy.mean(cv_image[:, :, 1])
        red_mean = numpy.mean(cv_image[:, :, 2])
        
        bgr_mean = (blue_mean + green_mean + red_mean) / 3      
        
        
        print bgr_mean
        print '===='
        
        cv2.imshow("Image window", cv_image)      

video_get()
rospy.init_node('image_get', anonymous=True)
rospy.spin()
root = tk.Tk()
root.geometry('300x200')
text = tk.Text(root, background='black', foreground='white', font=('Comic Sans MS', 12))
text.pack()
root.bind('<KeyPress>', onKeyPress)
root.mainloop()  
cv2.destroyAllWindows()
