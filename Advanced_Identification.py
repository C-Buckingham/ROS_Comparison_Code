#!/usr/bin/env python

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from upper_body_detector.msg import UpperBodyDetector
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from multiprocessing.pool import ThreadPool
# from sklearn import tree

tmp_array = np.array([0, 0, 0])
line_count = 0
number_of_classes = 0
choice = ""
data_array = np.array([0])
blue_data_list = []
green_data_list = []
red_data_list = []
class_number = []
hist_pool = []

with open("Output.txt", "r") as text_file:
    for line in text_file:
        if line == "=\n":
            number_of_classes += 1

print "Reading in data file...\n"

if number_of_classes > 0:
    with open('Output.txt', 'r') as text_file:
        data = text_file.read().replace('\n', '')
        data = data.replace('[', '')
        data = data.replace(']', '')
        data = data.split("=")
        data = np.asarray(data)
        data = filter(None, data)
        for x in range(0, number_of_classes):
            class_number.append(x)
            data[x] = data[x].split('+')
            data[x] = np.asarray(data[x])
            data[x] = filter(None, data[x])
            split_data = data[x]
            for y in range(0, len(data)):
                if y == 0:
                    blue_data_list.append(data[y])
                elif y == 1:
                    green_data_list.append(data[y])
                else:
                    red_data_list.append(data[y])

# hist_array = np.asarray((blue_data_list, green_data_list, red_data_list))
class_number = np.asarray(class_number)
hist_array = np.asarray(blue_data_list)

count = 0
combined_hist_values = [0, 0, 0]
base_image = []

print "Finished reading data file.", len(class_number), "class(es) found.\n"

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

    def colour_Matching(hist_pool, video_image):

        hsv_video_image = cv2.cvtColor(video_image, cv2.COLOR_BGR2HSV)

        list_location_counter = 0

        bgr_video_image_hist_list = []
        hsv_video_image_hist_list = []

        for z in range(0, len(hist_pool)):
            bgr_comparison_result = []
            hsv_comparison_result = []

            print z
            print "+++++"

            for a in range(0, 3):
                bgr_video_image_hist = cv2.calcHist([video_image], [a], None, [256], [1, 256])
                # combined_hist_values[z] = bgr_video_image_hist.astype(int)

                if a == 0:
                    blue = bgr_video_image_hist
                elif a == 1:
                    green = bgr_video_image_hist
                else:
                    red = bgr_video_image_hist

                bgr_comparison_result.append(
                    cv2.compareHist(bgr_video_image_hist, hist_pool[z][0][a], cv2.cv.CV_COMP_CORREL))

                if a < 2:

                    hsv_video_image_hist = cv2.calcHist([hsv_video_image], [a], None, [256], [1, 256])
                    if a == 0:
                        hue = hsv_video_image_hist
                    elif a == 1:
                        sat = hsv_video_image_hist

                    hsv_comparison_result.append(
                        cv2.compareHist(hsv_video_image_hist, hist_pool[z][1][a], cv2.cv.CV_COMP_CORREL))

            bgr_avg_correlation = np.mean(bgr_comparison_result)
            hsv_avg_correlation = np.mean(hsv_comparison_result)

            if bgr_avg_correlation > 0.85:
                cv2.imshow("Live Image", video_image)
                print "Match Found!"
                return bgr_video_image_hist_list, hsv_video_image_hist_list
            else:
                print "No Match Found"
                bgr_video_image_hist_list.append([])
                hsv_video_image_hist_list.append([])
                bgr_video_image_hist_list[list_location_counter].append(blue)
                bgr_video_image_hist_list[list_location_counter].append(green)
                bgr_video_image_hist_list[list_location_counter].append(red)
                hsv_video_image_hist_list[list_location_counter].append(hue)
                hsv_video_image_hist_list[list_location_counter].append(sat)
                list_location_counter += 1

        return bgr_video_image_hist_list, hsv_video_image_hist_list

    def feature_Matching(base_image, video_image):
        print "Feature Matching"

        orb = cv2.ORB()

        kp1, des1 = orb.detectAndCompute(video_image, None)
        kp2, des2 = orb.detectAndCompute(base_image, None)

        #                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        # matches = bf.knnMatch(des1, des2)

        #                    Why I am using ORB: https://www.willowgarage.com/sites/default/files/orb_final.pdf

        # good = []
        # for m,n in matches:
        #    if m.distance < 0.75*n.distance:
        #        good.append([m])

        # match = sorted(match, key = lambda x:x.distance)

        matches = sorted(matches, key=lambda m: m.distance)

        video_image = draw_matches(video_image, kp1, base_image, kp2, matches[:10])

        cv2.imshow("Image window", video_image)

        # video_image = cv2.drawKeypoints(video_image,kp1,color=(0, 255, 0), flags=0)
        # base_image = cv2.drawKeypoints(base_image, kp2, color=(0, 255, 0), flags=0)

    global options

    options = {'C': colour_Matching, 'F': feature_Matching}

    def image_callback(self, img, person, depth):
        global count
        global choice
        global base_image
        global hist_pool

        hist_pool_location = 0

        result = []
        video_image_list = []

        if len(person.height) > 0:
            for x in range(0, len(person.height)):

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

                count += 1

                depth_offset_percentage = float(person_x / 640.0 * 100.0)

                depth_image = depth_image.transpose()
                depth_image = np.roll(depth_image, 35)
                depth_image = depth_image.transpose()

                if depth_offset_percentage >= 70:
                    depth_image = np.roll(depth_image, -15)
                elif depth_offset_percentage <= 30:
                    depth_image = np.roll(depth_image, 15)

                depth_image = depth_image[person_y:(person_y + person_w) * 2, person_x:person_x + person_h]

                depth_image_shape = depth_image.shape
                depth_image = depth_image.flatten()
                depth_image = np.where(depth_image < person.median_depth[x] * 1.10, 1, 0)
                #            depth_image = np.where(depth_image > person.median_depth[x]*1.50, 1, 0)
                depth_image = np.reshape(depth_image, depth_image_shape)

                if count == 10 and person_h:
                    base_image = video_image[person_y:(person_y + person_w) * 2, person_x:person_x + person_h]
                    base_hsv = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)

                    tmp_bgr_hist_store = []
                    tmp_hsv_hist_store = []

                    for y in range(0, 3):
                        base_image[:, :, y] = base_image[:, :, y] * depth_image

                        tmp_bgr_hist_store.append(cv2.calcHist([base_image], [y], None, [256], [1, 256]))

                        if y < 2:
                            tmp_hsv_hist_store.append(cv2.calcHist([cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)],
                                                          [y], None, [256], [1, 256]))

                    hist_pool.append([])
                    hist_pool[0].append(tmp_bgr_hist_store)
                    hist_pool[0].append(tmp_hsv_hist_store)

                elif count == 10 and not person_h:
                    count = 0
                elif count > 10:
                    video_image = video_image[person_y:(person_y + person_w) * 2, person_x:person_x + person_h]
                    for y in range(0, 3):
                        video_image[:, :, y] = video_image[:, :, y] * depth_image

                    video_image_list.append(video_image)

                    print len(video_image_list)
                    cv2.imshow("Base Image", base_image)

                    print "Hist Pool Length: ", len(hist_pool)

                    print "============"

                    tmp = options[choice](hist_pool, video_image)

                    # if len(hist_pool) > 2:
                    #     print ""

                    if tmp[0] != []:
                        hist_pool_location += 1

                        hist_pool.append([])

                        hist_pool[hist_pool_location].append(tmp[0][0])
                        hist_pool[hist_pool_location].append(tmp[1][0])

                    # temp = []
                    # temph = []
                    #
                    # temp.append(blue)
                    # temp.append(green)
                    # temp.append(red)
                    #
                    # temph.append(hue)
                    # temph.append(sat)

                    # hist_pool[hist_pool_location].append(temp)
                    # hist_pool[hist_pool_location].append(temph)


person_comparison()
rospy.init_node('person_comparison', anonymous=True)
rospy.spin()
cv2.destroyAllWindows()

max_values = np.array([0])
max_values_index = np.array([0])

for x in range(0, len(combined_hist_values)):
    combined_hist_values[x] = map(int, combined_hist_values[x])
    if x == 0:
        max_values[x] = max(combined_hist_values[x])
        max_values_index[x] = combined_hist_values.index(max_values)
    elif x == 1:
        max_values[x] = max(combined_hist_values[x])
        max_values_index[x] = combined_hist_values.index(max_values[x])
    else:
        max_values[x] = max(combined_hist_values[x])
        max_values_index[x] = combined_hist_values.index(max_values[x])

with open("Output.txt", "a") as text_file:
    text_file.write("=\n")

for x in range(0, 3):
    with open("Output.txt", "a") as text_file:
        text_file.write("+")
        text_file.write("\n%s" % max_values[x])
        text_file.write("\n%s" % max_values_index[x])
        text_file.write("\n")