#!/usr/bin/env python

from sensor_msgs.msg import CompressedImage
import rospy
import roslib
import cv2 as cv
import numpy as np
import sys
import time
import json

import argparse

parser = argparse.ArgumentParser(description='Augmented Reality using Aruco markers in OpenCV')
parser.add_argument('--source', help='Path to image file.')
parser.add_argument('--destination', help='Path to video file.')
args = parser.parse_args()

outputFile = "video.avi"
if (args.destination):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)

    outputFile = args.destination
    print("Storing it as :", outputFile)

cap = cv.VideoCapture(outputFile)
size = (800,600) #(int(2*cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv.VideoWriter_fourcc('M','J','P','G')
vid_writer = cv.VideoWriter(outputFile, fourcc, 28, size)

inputTopic = '/owi535/camera1/image_raw/compressed'

class image_feature:

    def __init__(self):
        '''Initialize, ros subscriber'''

        # subscribed Topic
        self.subscriber = rospy.Subscriber( inputTopic, CompressedImage, self.callback, queue_size=1)
        print("subscribed to {}".format(inputTopic) )

    def callback(self, ros_data):
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''
        #if VERBOSE:
        #print('received image of type: "%s"' % ros_data.format)
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        #print(np_arr.shape)
        # image_np = cv.imdecode(np_arr, cv.CV_LOAD_IMAGE_COLOR)
        image_np = cv.imdecode(np_arr, cv.IMREAD_COLOR)  # OpenCV >= 3.0:
        #resized = cv.resize(image_np, size, interpolation = cv.INTER_AREA)
        #concatenatedOutput = cv.hconcat([image_np.astype(float), image_np]);
        vid_writer.write(image_np)        

def main(args):
    rospy.init_node('image_feature', anonymous=True)
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
