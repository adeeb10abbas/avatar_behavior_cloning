import rosbag
import rospy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from avatar_msgs.msg import PTIPacket
from rdda_interface.msg import RDDAPacket
from time import time

def callback(image_left, image_right, image_table, right_smarty_arm, left_smarty_arm, right_glove=None, left_glove=None):
    global callbacks_received
    rospy.loginfo("Callback triggered - writing synchronized messages to the output bag.")
    bag_out.write('/usb_cam_left/image_raw', image_left, image_left.header.stamp)
    bag_out.write('/usb_cam_right/image_raw', image_right, image_right.header.stamp)
    bag_out.write('/usb_cam_table/image_raw', image_table, image_table.header.stamp)
    bag_out.write('/right_smarty_arm_output', right_smarty_arm, right_smarty_arm.header.stamp)
    bag_out.write('/left_smarty_arm_output', left_smarty_arm, left_smarty_arm.header.stamp)
    assert(right_glove and left_glove)  # Ensure both glove data are available
    bag_out.write('/throttled_rdda_l_master_output', right_glove, right_glove.header.stamp)
    bag_out.write('/throttled_rdda_right_master_output', left_glove, left_glove.header.stamp)
    callbacks_received += 1

def main(input_bag_path, output_bag_path):
    global bag_out, callbacks_received
    rospy.init_node('timesync_node', anonymous=True)

    bag_out = rosbag.Bag(output_bag_path, 'w')
    publishers = {}
    callbacks_received = 0
    messages_published = 0

    ## Ignore this bit; not used anymore I throttle it via another rosnode at recording time now 
    topic_throttle_rate = {
        '/throttled_rdda_right_master_output': 1,  # Throttle to 100 Hz
        '/throttled_rdda_l_master_output': 1       # Throttle to 100 Hz
    }
    last_published_time = {topic: time() for topic in topic_throttle_rate}

    subscribers = [
        ('/usb_cam_left/image_raw', Image),
        ('/usb_cam_right/image_raw', Image),
        ('/usb_cam_table/image_raw', Image),
        ('/right_smarty_arm_output', PTIPacket),
        ('/left_smarty_arm_output', PTIPacket),
        ('/throttled_rdda_right_master_output', RDDAPacket),
        ('/throttled_rdda_l_master_output', RDDAPacket),
    ]

    # Setup subscribers and synchronizer
    subs = []
    for topic, msg_type in subscribers:
        if "rdda" in topic:
            queue_size = 25
        else:
            queue_size = 25
        pub = rospy.Publisher(topic, msg_type, queue_size=queue_size)
        publishers[topic] = pub
        sub = Subscriber(topic, msg_type)
        subs.append(sub)

    rospy.loginfo("Setting up the ApproximateTimeSynchronizer.")
    ats = ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.5)
    ats.registerCallback(callback)

    # Publish all messages from the bag
    with rosbag.Bag(input_bag_path, 'r') as bag_in:
        for topic, msg, t in bag_in.read_messages():
            if topic in topic_throttle_rate:
                current_time = time()
                if current_time - last_published_time[topic] < topic_throttle_rate[topic]:
                    continue  # Skip this message
                last_published_time[topic] = current_time

            if topic in publishers:
                publishers[topic].publish(msg)
                rospy.sleep(0.01)  # Small sleep to maintain timing
                messages_published += 1

    # Wait for all callbacks to be processed
    wait_start_time = rospy.Time.now()
    max_wait_duration = rospy.Duration(2) 
    while callbacks_received < messages_published:
        if rospy.Time.now() - wait_start_time > max_wait_duration:
            rospy.logwarn("Timeout reached, proceeding with shutdown despite unmatched callback count.")
            break
        rospy.loginfo("Waiting for all callbacks to be processed...")
        rospy.sleep(0.1)

    rospy.loginfo("All messages published and callbacks processed, initiating shutdown.")
    rospy.signal_shutdown("Completed processing all messages from bag.")

    rospy.spin()  # Wait for any remaining callbacks after shutdown signal

    rospy.loginfo("Shutting down - closing output bag file.")
    bag_out.close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        rospy.logerr("Usage: script.py <input_bag_file> <output_bag_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
