## Author : @adeeb10abbas
## Date : 2023-10-24
## Description : This file is used to build the workspace for the project
## LICENSE : MIT

workspace(name = "avatar_behavior_cloning")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "com_google_benchmark",
    branch = "main",
    remote = "https://github.com/google/benchmark",
)

git_repository(
    name = "com_google_googletest",
    branch = "main",
    remote = "https://github.com/google/googletest",
)

git_repository(
    name = "com_github_google_rules_install",
    commit = "e12fa95adda9fae107a22798acc2a3849d46f94a",
    remote = "https://github.com/google/bazel_rules_install",
)

load("@com_github_google_rules_install//:deps.bzl", "install_rules_dependencies")

install_rules_dependencies()

load("@com_github_google_rules_install//:setup.bzl", "install_rules_setup")

install_rules_setup()

############# DRAKE #############
DRAKE_TAG = "v1.22.0"
DRAKE_CHECKSUM = "78cf62c177c41f8415ade172c1e6eb270db619f07c4b043d5148e1f35be8da09"  # noqa

http_archive(
    name = "drake",
    sha256 = DRAKE_CHECKSUM,
    strip_prefix = "drake-{}".format(DRAKE_TAG.lstrip("v")),
    urls = [
        "https://github.com/RobotLocomotion/drake/archive/refs/tags/{}.tar.gz".format(DRAKE_TAG),  # noqa
    ],
)

new_git_repository(
    name = "tinyxml2",
    build_file = "//external:tinyxml2.BUILD",
    commit = "e45d9d16d430a3f5d3eee9fe40d5e194e1e5e63a",
    remote = "https://github.com/leethomason/tinyxml2.git",
    shallow_since = "1648934420 -0700",
)

http_archive(
    name = "qhull",
    urls = ["https://github.com/qhull/qhull/archive/HEAD.tar.gz"],
    build_file = "//external:qhull.BUILD",
)
new_git_repository(
    name = "lodepng",
    build_file = "lodepng.BUILD",
    commit = "5601b8272a6850b7c5d693dd0c0e16da50be8d8d",
    remote = "https://github.com/lvandeve/lodepng.git",
    shallow_since = "1641772872 +0100",
)
## End of MUJOCO support ##

load("@drake//tools/workspace:default.bzl", "add_default_workspace")
load("@drake//tools/workspace:github.bzl", "github_archive")

add_default_workspace()

##########################

##### DRAKE ROS #####
## Adding Bazel_ROS2_Rules for drake-ros stuff to work ##
DRAKE_ROS_commit = "14d52a30ead03ec021cae605e219a471cc618dc2"
DRAKE_ROS_sha256 = "6b24558181953bbac80d0b37dafa4e9c09f3b62527355c1f0e5c9189a54956b4"
## Ref: Eric Cousineau's awesome script - 
## https://github.com/EricCousineau-TRI/repro/blob/50c3f52c6b745f686bef9567568437dc609a7f91/bazel/bazel_hash_and_cache.py

git_repository(
    name = "bazel_ros2_rules",
    commit = DRAKE_ROS_commit,
    strip_prefix = "bazel_ros2_rules",
    remote = "https://github.com/RobotLocomotion/drake-ros.git",
)

git_repository(
    name = "drake_ros_repo",
    commit = DRAKE_ROS_commit,
    strip_prefix = "drake_ros",
    remote = "https://github.com/RobotLocomotion/drake-ros.git",
)

load("@bazel_ros2_rules//deps:defs.bzl", "add_bazel_ros2_rules_dependencies")
add_bazel_ros2_rules_dependencies()

load("@bazel_ros2_rules//ros2:defs.bzl", "ros2_local_repository")

ROS2_PACKAGES = [
    "action_msgs",
    "builtin_interfaces",
    "console_bridge_vendor",
    "rclcpp",
    "rclcpp_action",
    "rclpy",
    "ros2cli",
    "ros2cli_common_extensions",
    "rosidl_default_generators",
    "tf2_ros",
    "tf2_ros_py",
    "visualization_msgs",
    "rosidl_default_runtime",
] + [
    # These are possible RMW implementations. Uncomment one and only one to
    # change implementations
    #"rmw_cyclonedds_cpp",
    "rmw_fastrtps_cpp",
]

# Use ROS 2
ros2_local_repository(
    name = "ros2",
    workspaces = ["/opt/ros/rolling",],
    include_packages = ROS2_PACKAGES,
)

## Additional Libraries ## See list here: https://github.com/mjbots/bazel_deps
load("//tools/workspace:default.bzl", "add_default_repositories")
add_default_repositories()
load("@com_github_mjbots_bazel_deps//tools/workspace:default.bzl",
     bazel_deps_add = "add_default_repositories")
bazel_deps_add()

## All the Python Stuff ##
load("@rules_python//python:repositories.bzl", "python_register_toolchains")
python_register_toolchains(
    name = "python3_11",
    python_version = "3.11",
)

load("@python3_11//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
    name = "pip",
    requirements = "//:requirements.txt",
)
load("@pip//:requirements.bzl", "install_deps")
# Initialize repositories for all packages in requirements.txt.
install_deps()
## End Python Stuff ##

###### Adding Pytorch ######
load("//tools/workspace:default.bzl", "add_pytorch")
add_pytorch()
