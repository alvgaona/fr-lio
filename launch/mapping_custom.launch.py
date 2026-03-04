import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_path = get_package_share_directory('fast_lio')
    default_config_path = os.path.join(package_path, 'config')
    default_rviz_config_path = os.path.join(
        package_path, 'rviz', 'fastlio.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time')
    config_path = LaunchConfiguration('config_path')
    config_file = LaunchConfiguration('config_file')
    rviz_use = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    rigid_body_name = LaunchConfiguration('rigid_body_name')
    namespace = LaunchConfiguration('namespace')
    mocap_use = LaunchConfiguration('mocap')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    )
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='mid360.yaml',
        description='Config file'
    )
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Use RViz to monitor results'
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value=default_rviz_config_path,
        description='RViz config file path'
    )
    declare_rigid_body_name_cmd = DeclareLaunchArgument(
        'rigid_body_name', default_value='91',
        description='Mocap rigid body name'
    )
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace', default_value='',
        description='Namespace for all nodes and topics'
    )
    declare_mocap_cmd = DeclareLaunchArgument(
        'mocap', default_value='false',
        description='Launch mocap converter node'
    )

    lidar_accumulator_node = Node(
        package='fast_lio',
        executable='lidar_accumulator',
        namespace=namespace,
        parameters=[{
            'accumulate_count': 10,
            'input_topic': '/livox/lidar',
            'output_topic': '/livox/lidar_accumulated',
        }],
        output='screen'
    )

    livox_imu_to_base_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        namespace=namespace,
        arguments=['--frame-id', 'imu_link',
                   '--child-frame-id', 'base_link',
                   '--z', '-0.12'],
    )

    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        namespace=namespace,
        parameters=[PathJoinSubstitution([config_path, config_file]),
                    {'use_sim_time': use_sim_time}],
        output='screen'
    )

    mocap_converter_node = Node(
        package='fast_lio',
        executable='mocap_converter',
        namespace=namespace,
        parameters=[{
            'rigid_body_name': ParameterValue(rigid_body_name, value_type=str),
            'mocap_topic': '/mocap/rigid_bodies',
            'odom_frame': 'odom',
        }],
        output='screen',
        condition=IfCondition(mocap_use)
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(rviz_use)
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)
    ld.add_action(declare_rigid_body_name_cmd)
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_mocap_cmd)

    ld.add_action(lidar_accumulator_node)
    ld.add_action(livox_imu_to_base_link)
    ld.add_action(fast_lio_node)
    ld.add_action(mocap_converter_node)
    ld.add_action(rviz_node)

    return ld
