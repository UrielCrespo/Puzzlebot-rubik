from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ==========================================================
    # Launch arguments
    # ==========================================================

    video_device = LaunchConfiguration('video_device')
    debug = LaunchConfiguration('debug')
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

    # ==========================================================
    # Launch description
    # ==========================================================

    return LaunchDescription([

        # ------------------------------------------------------
        # Camera parameters
        # ------------------------------------------------------

        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video2',
            description='Logitech C920 video device'
        ),

        DeclareLaunchArgument(
            'debug',
            default_value='True',
            description='Enable debug image publishers'
        ),

        DeclareLaunchArgument(
            'rotate_image',
            default_value='False',
            description='Rotate image 180 degrees if camera is mounted upside down'
        ),

        # ------------------------------------------------------
        # Line detector parameters
        # ------------------------------------------------------

        DeclareLaunchArgument(
            'line_roi_top',
            default_value='0.45',
            description='Top limit of the lower ROI used for line detection'
        ),

        DeclareLaunchArgument(
            'trap_top_frac',
            default_value='0.08',
            description='Top width fraction of the triangular line mask'
        ),

        DeclareLaunchArgument(
            'trap_bottom_frac',
            default_value='0.65',
            description='Bottom width fraction of the triangular line mask'
        ),

        # ------------------------------------------------------
        # Traffic light detector parameters
        # ------------------------------------------------------

        DeclareLaunchArgument(
            'traffic_roi_bottom',
            default_value='0.72',
            description='Bottom limit of the traffic light search ROI'
        ),

        DeclareLaunchArgument(
            'traffic_max_center_y_frac',
            default_value='0.68',
            description='Maximum valid candidate center height; rejects floor reflections'
        ),

        DeclareLaunchArgument(
            'traffic_min_center_y_frac',
            default_value='0.03',
            description='Minimum valid candidate center height'
        ),

        DeclareLaunchArgument(
            'min_score_red',
            default_value='45.0',
            description='Minimum score required to accept RED'
        ),

        DeclareLaunchArgument(
            'min_score_yellow',
            default_value='18.0',
            description='Minimum score required to accept YELLOW'
        ),

        DeclareLaunchArgument(
            'min_score_green',
            default_value='35.0',
            description='Minimum score required to accept GREEN'
        ),

        # ------------------------------------------------------
        # USB camera node
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Main perception node
        # ------------------------------------------------------

        Node(
            package='traffic_light_vision',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{

                # General
                'debug': debug,
                'rotate_image': rotate_image,

                # Line detection
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

                # Traffic light processing
                'traffic_process_fps': 10.0,
                'traffic_roi_bottom': traffic_roi_bottom,

                # Reflection rejection
                'traffic_max_center_y_frac': traffic_max_center_y_frac,
                'traffic_min_center_y_frac': traffic_min_center_y_frac,

                # Traffic scores
                'min_score_red': min_score_red,
                'min_score_yellow': min_score_yellow,
                'min_score_green': min_score_green,

                # Traffic blob filtering
                'traffic_min_area': 12.0,
                'traffic_max_area': 6000.0,
                'traffic_min_circularity': 0.10,
                'traffic_min_fill_ratio': 0.12,
                'traffic_max_aspect_ratio': 3.2,

                # Dynamic ROI / tracking
                'roi_margin': 180,
                'max_lost_frames': 12,

                # Anti-flicker / temporal filter
                'yellow_hold_max': 16,
                'traffic_buffer_size': 10,
                'red_votes_required': 2,
                'yellow_votes_required': 2,
                'green_votes_required': 2,
                'unknown_votes_required': 7,

                # Debug
                'debug_fps': 5.0,
                'debug_jpeg_quality': 45,
            }]
        ),

        # ------------------------------------------------------
        # Debug republisher
        # Converts compressed debug images to raw images so
        # rqt_image_view can display them easily.
        # ------------------------------------------------------

        Node(
            package='traffic_light_vision',
            executable='debug_republisher',
            name='debug_republisher',
            output='screen'
        ),
    ])
