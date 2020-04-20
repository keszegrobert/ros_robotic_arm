#!/usr/bin/env python
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""
__author__ = 'Simon Haller <simon.haller at uibk.ac.at>'
__version__ = '0.1'
__license__ = 'BSD'
# Python libs
from sensor_msgs.msg import CompressedImage
import rospy
import roslib
import cv2
from scipy.ndimage import filters
import numpy as np
import sys
import time

# numpy and scipy

# OpenCV

# Ros libraries

# Ros Messages
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE = False


class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher(
            "/image_processor/image_raw/compressed",
            CompressedImage, queue_size=5)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber(
            "/owi535/camera1/image_raw/compressed",
            CompressedImage,
            self.callback,
            queue_size=1)
        if VERBOSE:
            print("subscribed to /owi535/camera1/image_raw/compressed")

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE:
            print('received image of type: "%s"' % ros_data.format)

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0:

        #### Feature detectors using CV2 ####
        # "","Grid","Pyramid" +
        # "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
        method = "GridFAST"
        #feat_det = cv2.FeatureDetector_create(method)
        feat_det = cv2.FastFeatureDetector_create()
        time1 = time.time()

        # convert np image to grayscale
        featPoints = feat_det.detect(
            cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY))
        time2 = time.time()
        if VERBOSE:
            print '%s detector found: %s points in: %s sec.' % (
                method, len(featPoints), time2-time1
            )

        cv2.circle(image_np, (100, 100), 3, (0, 0, 255), -1)
        for featpoint in featPoints:
            x, y = featpoint.pt
            cv2.circle(image_np, (int(x), int(y)), 3, (0, 0, 255), -1)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)

        # self.subscriber.unregister()


def main(args):
    rospy.init_node('image_feature', anonymous=True)
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
