import unittest
import numpy as np
from multiprocessing import Manager
from unittest.mock import patch, MagicMock
from diffusion_ros_interface import SubscriberNode
from sensor_msgs.msg import Image
from rdda_interface.msg import RDDAPacket
from avatar_msgs.msg import PTIPacket
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import time
from multiprocessing import Process, Manager
import logging

logging.basicConfig(filename='shared_obs_timestamps.log', level=logging.INFO, format='%(asctime)s - %(message)s')


manager = Manager()
shared_obs_dict = manager.dict()

def start_subscriber_node(shared_dict):
    node = SubscriberNode(shared_dict)
    # node.run()

subscriber_process = Process(target=start_subscriber_node, args=(shared_obs_dict,))
subscriber_process.start()

# subscriber_process = Process(target=SubscriberNode, args=(shared_obs_dict,))
# subscriber_process.start()    # self
# node.run()
# Configure logging

for _ in range(100):
    # current_dict = dict(shared_obs_dict)
    # for key, value in current_dict.items():
    logging.info(f"Timestamp: {shared_obs_dict}")
    time.sleep(0.1)  # Reduce the frequency of prints

subscriber_process.join()  # Ensure the subscriber process has completed