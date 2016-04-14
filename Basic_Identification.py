#!/usr/bin/env python

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

tmp_array = np.array([0, 0, 0])
line_count = 0
number_of_classes = 0
choice = ""
data_array = np.array([0])
count = 0
base_image = []
base_hsv = []
base_image_hist = []
base_hsv_hist = []

while choice != 'F' and choice != 'C':
    choice = raw_input('Choose between Feature Matching (F) or Colour Matching (C): ')
    if choice != 'F' and choice != 'C':
        print "You must enter either ""C"" for Colour Matching or ""F"" for feature matching."

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
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

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


class person_comparison:

    def __init__(self):
        
        # Create new windows
        cv2.namedWindow("Live Image", 1)
        cv2.namedWindow("Base Image", 1)
        cv2.namedWindow("Original Image", 1)
        cv2.startWindowThread()
        self.bridge = CvBridge()

        # Commented code below used for testing only
        # self.match_pub = rospy.Publisher("~match_image_out", Image, queue_size=1)
        # self.base_pub = rospy.Publisher("~base_image_out", Image, queue_size=1)
        # self.match_string = rospy.Publisher("~match_bool", String, queue_size=1)

        # Subscribe to the required topics. Using /camera/ instead of /head_xiton/ when
        # not running on Linda
        image_sub = Subscriber(
            "/head_xtion/rgb/image_color",
            Image,
            queue_size=1
        )

        person_sub = Subscriber(
            "/upper_body_detector/detections",
            UpperBodyDetector,
            queue_size=1
        )

        depth_sub = Subscriber(
            "/head_xtion/depth/image_rect",
            Image,
            queue_size=1
        )
        
        # Time syncronizer is implimented to make sure that all of the frames match up from all of the topics.
        ts = ApproximateTimeSynchronizer([image_sub, person_sub, depth_sub], 1, 0.1)
        ts.registerCallback(self.image_callback)

    def colour_Matching(self, base_image_hist, base_hsv_hist, video_image):

        hsv_video_image = cv2.cvtColor(video_image, cv2.COLOR_BGR2HSV)
        bgr_comparison_result = []
        hsv_comparison_result = []

        # Histogram calculation loop.
        for a in range(0, 3):
            # BGR histogram calculation
            bgr_video_image_hist = cv2.calcHist([video_image], [a], None, [256], [1, 256])
            
            # Correlation calculation and storing for BGR image happens here
            bgr_comparison_result.append(
                cv2.compareHist(bgr_video_image_hist, base_image_hist, cv2.cv.CV_COMP_CORREL))

            if a < 2:
                # HSV histogram calculation
                hsv_video_image_hist = cv2.calcHist([hsv_video_image], [a], None, [256], [1, 256])
                
                # Correlation calculation and storing for HSV image happens here
                hsv_comparison_result.append(
                    cv2.compareHist(hsv_video_image_hist, base_hsv_hist, cv2.cv.CV_COMP_CORREL))

        # Average the correlation results
        bgr_avg_correlation = np.mean(bgr_comparison_result)
        hsv_avg_correlation = np.mean(hsv_comparison_result)

        if bgr_avg_correlation > 0.85 or hsv_avg_correlation > 0.80:
            
            # Display image match
            cv2.imshow("Live Image", video_image)
            
            # Commented code below used for testing only
            # self.match_pub.publish(self.bridge.cv2_to_imgmsg(video_image, "bgr8"))
            
            print "Match Found!"
        else:
            print "No Match Found"

    def feature_Matching(base_image, video_image):
        # This part of the code does not work, was an ealry attempt to get some feture matching working.

        # ORB feature detector is created here.
        orb = cv2.ORB()

        # Compute and store the ORB features.
        kp1, des1 = orb.detectAndCompute(video_image, None)
        kp2, des2 = orb.detectAndCompute(base_image, None)

        # Finds the matching sections between the two images.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        matches = bf.match(des1, des2)

        # Sort the matches into the highest accuracy.
        matches = sorted(matches, key=lambda m: m.distance)

        # Call the draw matches function to draw the 10 closest matches.
        video_image = draw_matches(video_image, kp1, base_image, kp2, matches[:10])

        cv2.imshow("Image window", video_image)

        # video_image = cv2.drawKeypoints(video_image,kp1,color=(0, 255, 0), flags=0)
        # base_image = cv2.drawKeypoints(base_image, kp2, color=(0, 255, 0), flags=0)

    global options

    # Declaring the dictionary
    options = {'C': colour_Matching, 'F': feature_Matching}

    def image_callback(self, img, person, depth):
        global count
        global choice
        global base_image
        global base_hsv
        global base_image_hist
        global base_hsv_hist

        if len(person.height) > 0:
            for x in range(0, len(person.height)):

                # Convert the messages from the sensor into an image
                try:
                    video_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
                    depth_image = self.bridge.imgmsg_to_cv2(depth)
                except CvBridgeError, e:
                    print e

                cv2.imshow("Original Image", video_image)

                # Making sure that the box stays within the bounds of the image
                person_h = min(person.height[x], 480)
                person_w = min(person.width[x], 640)
                person_x = min(person.pos_x[x], 480)
                person_y = min(person.pos_y[x], 640)

                person_h = max(person.height[x], 0)
                person_w = max(person.width[x], 0)
                person_x = max(person.pos_x[x], 0)
                person_y = max(person.pos_y[x], 0)

                count += 1

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

                if count == 10 and person_h:
                    # Crop the base image
                    base_image = video_image[person_y:(person_y + person_w) * 2, person_x:person_x + person_h]
                    
                    # Convert the image to HSV
                    base_hsv = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)

                    # Loop to go through and apply the mask to the base image
                    for y in range(0, 3):
                        base_image[:, :, y] = base_image[:, :, y] * depth_image
                        base_image_hist = cv2.calcHist([base_image], [y], None, [256], [1, 256])
                        if y < 2:
                            base_hsv[:, :, y] = base_hsv[:, :, y] * depth_image
                            base_hsv_hist = cv2.calcHist([base_hsv], [y], None, [256], [1, 256])

                # Check to make sure that there is a person in the image before moving on
                elif count == 10 and not person_h:
                    count = 0
                elif count > 10:
                    
                    # Crop the video image.
                    video_image = video_image[person_y:(person_y + person_w) * 2, person_x:person_x + person_h]
                    
                    # Apply the mask to the image.
                    for y in range(0, 3):
                        video_image[:, :, y] = video_image[:, :, y] * depth_image

                    cv2.imshow("Base Image", base_image)
                    
                    # Commented code below used for testing only.
                    # self.base_pub.publish(self.bridge.cv2_to_imgmsg(base_image, "bgr8"))

                    # Search through the dictionary for the choice that the use selected and call that function.
                    options[choice](self, base_image_hist, base_hsv_hist, video_image)

rospy.init_node('person_comparison')
person_comparison()
rospy.spin()
cv2.destroyAllWindows()
