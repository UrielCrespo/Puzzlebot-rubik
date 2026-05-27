from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    video_device = LaunchConfiguration('video_device')
    rotate_image = LaunchConfiguration('rotate_image')

    line_roi_top = LaunchConfiguration('line_roi_top')
    trap_top_frac = LaunchConfiguration('trap_top_frac')
    trap_bottom_frac = LaunchConfiguration('trap_bottom_frac')

    traffic_roi_bottom = LaunchConfiguration('traffic_roi_bottom')
    traffic_max_center_y_frac = LaunchConfiguration('traffic_max_center_y_frac')
    traffic_min_center_y_frac = LaunchConfiguration('traffic_min_center_y_frac')

    min_score_red = LaunchConfiguration('min_score_red')
    min_score_yellow = LaunchConfiguration('min_score_yellow')
    min_score_green = LaunchConfiguration('min_score_green')

    return LaunchDescription([

        DeclareLaunchArgument('video_device', default_value='/dev/video2'),
        DeclareLaunchArgument('rotate_image', default_value='False'),

        DeclareLaunchArgument('line_roi_top', default_value='0.45'),
        DeclareLaunchArgument('trap_top_frac', default_value='0.08'),
        DeclareLaunchArgument('trap_bottom_frac', default_value='0.65'),

        DeclareLaunchArgument('traffic_roi_bottom', default_value='0.72'),
        DeclareLaunchArgument('traffic_max_center_y_frac', default_value='0.68'),
        DeclareLaunchArgument('traffic_min_center_y_frac', default_value='0.03'),

        DeclareLaunchArgument('min_score_red', default_value='45.0'),
        DeclareLaunchArgument('min_score_yellow', default_value='18.0'),
        DeclareLaunchArgument('min_score_green', default_value='35.0'),

        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            output='screen',
            parameters=[{
                'video_device': video_device,
                'image_width': 640,
                'image_height': 480,
                'framerate': 30.0,
                'pixel_format': 'mjpeg2rgb',
                'camera_name': 'logitech_c920'
            }]
        ),

        Node(
            package='traffic_light_vision',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{
                'debug': False,
                'rotate_image': rotate_image,

                'line_resize_width': 320,
                'line_resize_height': 240,
                'line_roi_top': line_roi_top,
                'blur_kernel': 7,
                'morph_kernel': 5,
                'min_area': 400,
                'max_area': 100000,
                'score_distance_weight': 8.0,
                'lookahead_row': 0.25,
                'lost_timeout': 1.95,
                'trap_top_frac': trap_top_frac,
                'trap_bottom_frac': trap_bottom_frac,

                'traffic_process_fps': 10.0,
                'traffic_roi_bottom': traffic_roi_bottom,
                'traffic_max_center_y_frac': traffic_max_center_y_frac,
                'traffic_min_center_y_frac': traffic_min_center_y_frac,

                'min_score_red': min_score_red,
                'min_score_yellow': min_score_yellow,
                'min_score_green': min_score_green,

                'traffic_min_area': 12.0,
                'traffic_max_area': 6000.0,
                'traffic_min_circularity': 0.10,
                'traffic_min_fill_ratio': 0.12,
                'traffic_max_aspect_ratio': 3.2,

                'roi_margin': 180,
                'max_lost_frames': 12,

                'yellow_hold_max': 16,
                'traffic_buffer_size': 10,
                'red_votes_required': 2,
                'yellow_votes_required': 2,
                'green_votes_required': 2,
                'unknown_votes_required': 7,

                'debug_fps': 5.0,
                'debug_jpeg_quality': 45,
            }]
        ),
    ])
