#!/usr/bin/env python  
import rospy
import math
import tf2_ros
from std_msgs.msg import Float64
import json

def logCamPos():
    pass


def getMarkerState(tfBuffer, marker):
    trans = None
    rot = None
    counter = 0
    rate = rospy.Rate(100) # 100hz
    while not rospy.is_shutdown():
        counter += 1
        if counter > 10:
            print('Marker '+marker+' not found')
            break
        try:
            now = rospy.Time.now() 
            #listener.waitForTransform(marker, "camera_link_optical", now, rospy.Duration(10.0))
            trans = tfBuffer.lookup_transform(marker, "camera_link_optical", rospy.Time(0), rospy.Duration(1.0))
            with open('src/transformations.txt', 'a') as f:
                towrite = {
                    'transx': trans.transform.translation.x,
                    'transy': trans.transform.translation.y,
                    'transz': trans.transform.translation.z,
                    'rotx': trans.transform.rotation.x,
                    'roty': trans.transform.rotation.y,
                    'rotz': trans.transform.rotation.z,
                    'rotw': trans.transform.rotation.w,
                    'time': now.to_sec(),
                    'marker': marker,
                }
                f.write(json.dumps(towrite))
                f.write('\n')

            return trans
        except tf2_ros.LookupException as e:
            print('lookup exceptions')
        except tf2_ros.ConnectivityException as e:
            print('connectivity exceptions')
        except tf2_ros.ExtrapolationException as e:
            print('extrapolation exceptions')
        except tf2_ros.TransformException as e:
            print('transform exceptions')
        rate.sleep()
        continue
    return None

def moveJoint(command, data):
    # rostopic pub -1 /owi535/rotationbase_position_controller/command std_msgs/Float64 0.1
    print('Moving the joint')
    pub = rospy.Publisher(command, Float64, queue_size=10)
    msg = Float64()
    msg.data = data
    rate = rospy.Rate(10) # 10hz
    counter = 0
    try:
        pub.publish(msg)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        return
    rate.sleep()

def main():
    rospy.init_node('arm_recognizer')
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    trans = getMarkerState(tfBuffer, 'marker_id23')
    print('trans:'+str(trans))
    for angle in range(1, 900):
        moveJoint('/owi535/rotationbase_position_controller/command', angle/1000.0)
        trans = getMarkerState(tfBuffer, 'marker_id23')
        logCamPos()
    for angle in range(1, 900):
        moveJoint('/owi535/rotationbase_position_controller/command', 0.9 - angle/1000.0)
        trans = getMarkerState(tfBuffer, 'marker_id23')
        logCamPos()
    for angle in range(1, 900):
        moveJoint('/owi535/rotationbase_position_controller/command', -angle/1000.0)
        trans = getMarkerState(tfBuffer, 'marker_id23')
        logCamPos()
    for angle in range(1, 900):
        moveJoint('/owi535/rotationbase_position_controller/command', -0.9 + angle/1000.0)
        trans = getMarkerState(tfBuffer, 'marker_id23')
        logCamPos()

    moveJoint('/owi535/rotationbase_position_controller/command', 0.0)
    

if __name__ == '__main__':
    main()