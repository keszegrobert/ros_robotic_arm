#!/usr/bin/env python

from sensor_msgs.msg import CompressedImage
import rospy
import roslib
import cv2
from cv2 import aruco
from scipy.ndimage import filters
import numpy as np
import sys
import time
inToM = 0.0254

# Camera calibration info
maxWidthIn = 17
maxHeightIn = 23
maxWidthM = maxWidthIn * inToM
maxHeightM = maxHeightIn * inToM

charucoNSqVert = 10
charucoSqSizeM = float(maxHeightM) / float(charucoNSqVert)
charucoMarkerSizeM = charucoSqSizeM * 0.7
# charucoNSqHoriz = int(maxWidthM / charucoSqSizeM)
charucoNSqHoriz = 16

charucoDictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

markerSizeIn = 5
markerSizeM = markerSizeIn * inToM

detectorParams = aruco.DetectorParameters_create()
detectorParams.cornerRefinementMaxIterations = 500
detectorParams.cornerRefinementMinAccuracy = 0.001
detectorParams.adaptiveThreshWinSizeMin = 3
detectorParams.adaptiveThreshWinSizeMax = 230
detectorParams.adaptiveThreshWinSizeStep = 10
detectorParams.maxMarkerPerimeterRate = 0.5
detectorParams.minMarkerPerimeterRate = 0.05

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

        time1 = time.time()

        # convert np image to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        detectorParameters = aruco.DetectorParameters_create()
        marker_corners, marker_ids, _ = aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=detectorParameters
        )

        '''marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray,
            charucoDictionary,
            parameters=detectorParams
        )'''

        cv2.circle(image_np, (100, 100), 3, (0, 0, 255), -1)
        if len(marker_corners) > 0:
            aruco.drawDetectedMarkers(image_np, marker_corners, marker_ids)
            print("!!! Found ", len(marker_corners), " Markers !!!")
        else:
            print("!!! Markers not detected !!!")

        #### Create CompressedImage ####
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
