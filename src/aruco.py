#!/usr/bin/env python

from sensor_msgs.msg import CompressedImage
import rospy
import roslib
import cv2
from cv2 import aruco
# from scipy.ndimage import filters
import numpy as np
import sys
import time
import json

import random
# inToM = 0.0254

# Camera calibration info
# maxWidthIn = 17
# maxHeightIn = 23
# maxWidthM = maxWidthIn * inToM
# maxHeightM = maxHeightIn * inToM

# charucoNSqVert = 10
# charucoSqSizeM = float(maxHeightM) / float(charucoNSqVert)
# charucoMarkerSizeM = charucoSqSizeM * 0.7
# charucoNSqHoriz = int(maxWidthM / charucoSqSizeM)
# charucoNSqHoriz = 16

# charucoDictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# markerSizeIn = 2
# markerSizeM = markerSizeIn * inToM


def generateDetectorParams():
    detectorParams = aruco.DetectorParameters_create()
    detectorParams.adaptiveThreshWinSizeMin = random.randrange(4, 100)
    detectorParams.adaptiveThreshWinSizeMax = random.randrange(100, 500)
    detectorParams.adaptiveThreshWinSizeStep = random.randrange(
        1, 100)  # maybe more
    detectorParams.adaptiveThreshConstant = random.randrange(0, 14)
    detectorParams.minMarkerPerimeterRate = random.random()*0.30  # reviewed
    detectorParams.maxMarkerPerimeterRate = random.random()*20  # maybe more
    detectorParams.polygonalApproxAccuracyRate = random.random()/10.0
    detectorParams.minCornerDistanceRate = random.random()/10.0
    detectorParams.minDistanceToBorder = random.randrange(1, 10)
    detectorParams.minMarkerDistanceRate = random.random()/10.0
    detectorParams.doCornerRefinement = (random.random() < 0.5)
    if (detectorParams.doCornerRefinement):
        detectorParams.cornerRefinementWinSize = random.randrange(
            1, 10)  # 5 korol van sok
        detectorParams.cornerRefinementMaxIterations = random.randrange(
            1, 500)  # 30 korul van sok
        detectorParams.cornerRefinementMinAccuracy = random.random()/50.0 + \
            0.001  # vmi nem kerek
    detectorParams.minOtsuStdDev = random.randrange(0, 10)

    return detectorParams


def detectorParams2():
    detectorParams = aruco.DetectorParameters_create()
    detectorParams.adaptiveThreshWinSizeMin = 76
    detectorParams.adaptiveThreshWinSizeMax = 751
    detectorParams.adaptiveThreshWinSizeStep = 95
    detectorParams.adaptiveThreshConstant = 7
    detectorParams.minMarkerPerimeterRate = 0.084
    detectorParams.maxMarkerPerimeterRate = 7.4058
    detectorParams.polygonalApproxAccuracyRate = 0.03
    detectorParams.minCornerDistanceRate = 0.05
    detectorParams.minDistanceToBorder = 3
    detectorParams.minMarkerDistanceRate = 0.05
    detectorParams.doCornerRefinement = True
    detectorParams.cornerRefinementWinSize = 5
    detectorParams.cornerRefinementMaxIterations = 85
    detectorParams.cornerRefinementMinAccuracy = 0.001
    detectorParams.minOtsuStdDev = 5
    return detectorParams


def detectorParams23():
    detectorParams = aruco.DetectorParameters_create()
    detectorParams.adaptiveThreshWinSizeMin = 78
    detectorParams.adaptiveThreshWinSizeMax = 814
    detectorParams.adaptiveThreshWinSizeStep = 45
    detectorParams.adaptiveThreshConstant = 7
    detectorParams.minMarkerPerimeterRate = 0.052
    detectorParams.maxMarkerPerimeterRate = 4.399
    detectorParams.polygonalApproxAccuracyRate = 0.03
    detectorParams.minCornerDistanceRate = 0.05
    detectorParams.minDistanceToBorder = 3
    detectorParams.minMarkerDistanceRate = 0.05
    detectorParams.doCornerRefinement = True
    detectorParams.cornerRefinementWinSize = 5
    detectorParams.cornerRefinementMaxIterations = 85
    detectorParams.cornerRefinementMinAccuracy = 0.001
    detectorParams.minOtsuStdDev = 5
    return detectorParams


def saveresults(marker_corners, marker_ids, detectorParams):

    params = {
        "adThreshWinSizeMin": detectorParams.adaptiveThreshWinSizeMin,
        "adThreshWinSizeMax": detectorParams.adaptiveThreshWinSizeMax,
        "adThreshWinSizeStep": detectorParams.adaptiveThreshWinSizeStep,
        "adTreshConst": detectorParams.adaptiveThreshConstant,
        "minMarkerPerimeterRate": detectorParams.minMarkerPerimeterRate,
        "minMarkerDistanceRate": detectorParams.minMarkerDistanceRate,
        "maxMarkerPerimeterRate": detectorParams.maxMarkerPerimeterRate,
        "polyAppAccRate": detectorParams.polygonalApproxAccuracyRate,
        "doCornerRef": detectorParams.doCornerRefinement,
        "cornerRefWinSize": detectorParams.cornerRefinementWinSize,
        "cornerRefMaxIt": detectorParams.cornerRefinementMaxIterations,
        "cornerRefMinAcc": detectorParams.cornerRefinementMinAccuracy,
        "minCornerDistRate": detectorParams.minCornerDistanceRate,
        "minDistToBorder": detectorParams.minDistanceToBorder,
        "minOtsuStdDev": detectorParams.minOtsuStdDev,
    }
    res = []
    if len(marker_corners) == 1:
        res = [int(marker_ids[0][0])]
    if len(marker_corners) == 2:
        res = [int(marker_ids[0][0]), int(marker_ids[1][0])]
    if len(marker_corners) == 3:
        res = [int(marker_ids[0][0]), int(
            marker_ids[1][0]), int(marker_ids[2][0])]
    params["r"] = res
    towrite = json.dumps(params)

    with open("src/findings3.txt", "a") as f:
        print(towrite)
        f.write(towrite)
        f.write(',\n')


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
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0:

        # time1 = time.time()

        # convert np image to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
        # detectorParameters = aruco.DetectorParameters_create()
        detectorParameters = generateDetectorParams()
        marker_corners, marker_ids, _ = aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=detectorParameters
        )
        '''detectorParams = generateDetectorParams()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray,
            charucoDictionary,
            parameters=detectorParams
        )'''

        cv2.circle(image_np, (100, 100), 3, (0, 0, 255), -1)
        saveresults(marker_corners, marker_ids, detectorParameters)
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
