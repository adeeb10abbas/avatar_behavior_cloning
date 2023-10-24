#include <gtest/gtest.h>

#include <drake_ros/core/ros_interface_system.h>
#include <drake_ros/core/drake_ros.h>
#include <drake_ros/viz/rviz_visualizer.h>

using drake_ros::core::RosInterfaceSystem;
using drake_ros::core::DrakeRos;
using drake_ros::viz::RvizVisualizer;

TEST(DrakeRosTest, RosInitializationTest) {
    // Check if drake_ros initializes without any errors.
    ASSERT_NO_THROW(drake_ros::core::init());
}

TEST(DrakeRosTest, RosInterfaceSystemTest) {
    // Instantiate DrakeRos with the given node name.
    auto drake_ros_instance = std::make_unique<DrakeRos>("test_drake_ros_interface");

    // Check if RosInterfaceSystem instantiation works.
    ASSERT_NO_THROW(auto ros_interface_system = std::make_unique<RosInterfaceSystem>(std::move(drake_ros_instance)));
}

TEST(DrakeRosTest, RvizVisualizerTest) {
    // Instantiate DrakeRos with the given node name.
    auto drake_ros_rviz = std::make_unique<DrakeRos>("test_drake_ros_rviz");

    auto ros_interface_system_rviz = std::make_unique<RosInterfaceSystem>(std::move(drake_ros_rviz));
    
    // Check if RvizVisualizer can be created using the RosInterfaceSystem.
    ASSERT_NO_THROW(auto rviz_visualizer = ros_interface_system_rviz->get_ros_interface());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    drake_ros::core::init();
    return RUN_ALL_TESTS();
}
