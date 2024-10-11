import rospy
from diffusion_rollout_ros_sub import SubscriberNode
from collections import defaultdict

import rospy
import time
from collections import defaultdict

def main():
    # rospy.init_node('observation_subscriber_main')
    shared_obs_dict = defaultdict(lambda: None)
    subscriber = SubscriberNode(shared_obs_dict)

    try:
        while not rospy.is_shutdown():
            last_two_obs = subscriber.get_obs()
            if len(last_two_obs) == 2:
                print("Last two observations:")
                for idx, obs in enumerate(last_two_obs):
                    print(f"Observation {idx+1}:")
                    for key, value in obs.items():
                        print(f"  {key}: {type(value)}")
                # Sleep for a while before checking again
                time.sleep(1)
            else:
                print("Waiting for observations...")
                time.sleep(0.5)  # Adjust the sleep time as needed

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
