#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#img = cv2.imread('/home/chris/Downloads/Gaming-PC.jpg', 0)
#
#orb = cv2.ORB()
#
#kp = orb.detect(img,None)
#
#kp, des = orb.compute(img, kp)
#
#img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
#plt.imshow(img2),plt.show()

def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = (0, 255, 1)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    
    return new_img

class image_converter:
    
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
        base_image  = cv2.imread('/home/chris/catkin_ws/src/ROS_Comparison_Code/Base.jpg')    
        
        orb = cv2.ORB()

        kp1, des1 = orb.detectAndCompute(video_image,None)
        kp2, des2 = orb.detectAndCompute(base_image, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        matches = bf.match(des1,des2)

        matches = sorted(matches, key = lambda x:x.distance)
        
        video_image = draw_matches(video_image,kp1,base_image,kp2,matches[:10])
        
#        kp1, des = orb.compute(video_image, kp1)

#        video_image = cv2.drawKeypoints(video_image,kp1,color=(0,255,0), base_image,kp2,color(0,255,0), flags=0)
        
        cv2.imshow("Image window", video_image)  
        
image_converter()
rospy.init_node('image_converter', anonymous=True)
rospy.spin()
cv2.destroyAllWindows()