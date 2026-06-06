from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ==========================================================
    # Launch configurations
    # ==========================================================

    video_device = LaunchConfiguration('video_device')
    debug = LaunchConfiguration('debug')
    rotate_image = LaunchConfiguration('rotate_image')

    line_roi_top = LaunchConfiguration('line_roi_top')
    lookahead_row = LaunchConfiguration('lookahead_row')
    trap_top_frac = LaunchConfiguration('trap_top_frac')
    trap_bottom_frac = LaunchConfiguration('trap_bottom_frac')
    trap_top_row_frac = LaunchConfiguration('trap_top_row_frac')

    traffic_roi_x_min_frac = LaunchConfiguration('traffic_roi_x_min_frac')
    traffic_roi_x_max_frac = LaunchConfiguration('traffic_roi_x_max_frac')
    traffic_roi_y_min_frac = LaunchConfiguration('traffic_roi_y_min_frac')
    traffic_roi_y_max_frac = LaunchConfiguration('traffic_roi_y_max_frac')

    traffic_min_center_x_frac = LaunchConfiguration('traffic_min_center_x_frac')
    traffic_max_center_x_frac = LaunchConfiguration('traffic_max_center_x_frac')
    traffic_min_center_y_frac = LaunchConfiguration('traffic_min_center_y_frac')
    traffic_max_center_y_frac = LaunchConfiguration('traffic_max_center_y_frac')

    min_score_red = LaunchConfiguration('min_score_red')
    min_score_yellow = LaunchConfiguration('min_score_yellow')
    min_score_green = LaunchConfiguration('min_score_green')

    # ==========================================================
    # Launch description
    # ==========================================================

    return LaunchDescription([

        # ------------------------------------------------------
        # General arguments
        # ------------------------------------------------------

        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video2',
            description='Logitech camera video device'
        ),

        DeclareLaunchArgument(
            'debug',
            default_value='True',
            description='Enable debug image publishers'
        ),

        DeclareLaunchArgument(
            'rotate_image',
            default_value='False',
            description='Rotate image 180 degrees if needed'
        ),

        # ------------------------------------------------------
        # Line detector arguments
        # ------------------------------------------------------

        DeclareLaunchArgument(
            'line_roi_top',
            default_value='0.45',
            description='Top limit of the lower ROI used for line detection'
        ),

        DeclareLaunchArgument(
            'lookahead_row',
            default_value='0.55',
            description='Lookahead row inside line ROI. Higher value moves orange line down'
        ),

        DeclareLaunchArgument(
            'trap_top_frac',
            default_value='0.30',
            description='Top width fraction of the trapezoid mask'
        ),

        DeclareLaunchArgument(
            'trap_bottom_frac',
            default_value='0.65',
            description='Bottom width fraction of the trapezoid mask'
        ),

        DeclareLaunchArgument(
            'trap_top_row_frac',
            default_value='0.45',
            description='Vertical start of the trapezoid inside the line ROI'
        ),

        # ------------------------------------------------------
        # Traffic light ROI arguments
        # ------------------------------------------------------

        DeclareLaunchArgument(
            'traffic_roi_x_min_frac',
            default_value='0.75',
            description='Traffic ROI left boundary as image fraction'
        ),

        DeclareLaunchArgument(
            'traffic_roi_x_max_frac',
            default_value='1.00',
            description='Traffic ROI right boundary as image fraction'
        ),

        DeclareLaunchArgument(
            'traffic_roi_y_min_frac',
            default_value='0.00',
            description='Traffic ROI top boundary as image fraction'
        ),

        DeclareLaunchArgument(
            'traffic_roi_y_max_frac',
            default_value='0.50',
            description='Traffic ROI bottom boundary as image fraction'
        ),

        DeclareLaunchArgument(
            'traffic_min_center_x_frac',
            default_value='0.75',
            description='Minimum valid traffic candidate x center'
        ),

        DeclareLaunchArgument(
            'traffic_max_center_x_frac',
            default_value='1.00',
            description='Maximum valid traffic candidate x center'
        ),

        DeclareLaunchArgument(
            'traffic_min_center_y_frac',
            default_value='0.00',
            description='Minimum valid traffic candidate y center'
        ),

        DeclareLaunchArgument(
            'traffic_max_center_y_frac',
            default_value='0.52',
            description='Maximum valid traffic candidate y center'
        ),

        # ------------------------------------------------------
        # Traffic score arguments
        # ------------------------------------------------------

        DeclareLaunchArgument(
            'min_score_red',
            default_value='55.0'
        ),

        DeclareLaunchArgument(
            'min_score_yellow',
            default_value='28.0'
        ),

        DeclareLaunchArgument(
            'min_score_green',
            default_value='45.0'
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
                'score_distance_weight': 12.0,
                'lookahead_row': lookahead_row,
                'lost_timeout': 1.95,
                'trap_top_frac': trap_top_frac,
                'trap_bottom_frac': trap_bottom_frac,
                'trap_top_row_frac': trap_top_row_frac,

                # Traffic light processing
                'traffic_process_fps': 12.0,

                # Traffic ROI
                'traffic_roi_x_min_frac': traffic_roi_x_min_frac,
                'traffic_roi_x_max_frac': traffic_roi_x_max_frac,
                'traffic_roi_y_min_frac': traffic_roi_y_min_frac,
                'traffic_roi_y_max_frac': traffic_roi_y_max_frac,

                # Kept for compatibility
                'traffic_roi_bottom': 0.72,

                # Candidate validation
                'traffic_min_center_x_frac': traffic_min_center_x_frac,
                'traffic_max_center_x_frac': traffic_max_center_x_frac,
                'traffic_min_center_y_frac': traffic_min_center_y_frac,
                'traffic_max_center_y_frac': traffic_max_center_y_frac,

                # Traffic scores
                'min_score_red': min_score_red,
                'min_score_yellow': min_score_yellow,
                'min_score_green': min_score_green,

                # Brightness filtering
                'traffic_bright_percentile': 92.0,
                'traffic_min_v_floor': 95.0,
                'traffic_min_candidate_mean_v': 105.0,
                'traffic_min_candidate_peak_v': 145.0,
                'traffic_min_candidate_bright_ratio': 0.20,

                # Blob filtering
                'traffic_min_area': 12.0,
                'traffic_max_area': 3500.0,
                'traffic_min_circularity': 0.12,
                'traffic_min_fill_ratio': 0.14,
                'traffic_max_aspect_ratio': 3.2,

                # Tracking / temporal state
                'roi_margin': 180,
                'max_lost_frames': 12,

                # Anti-flicker
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
        # Converts compressed debug images to raw images for rqt_image_view.
        # ------------------------------------------------------

        Node(
            package='traffic_light_vision',
            executable='debug_republisher',
            name='debug_republisher',
            output='screen'
        ),

	Node(
            package='SignalDetector',
            executable='detector_node',
            name='detector_senales',
            output='screen',
        ),
    ])
