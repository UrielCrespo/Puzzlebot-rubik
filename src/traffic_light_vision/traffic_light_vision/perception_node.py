#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray, String


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        self.bridge = CvBridge()

        # ==========================================================
        # General parameters
        # ==========================================================
        self.declare_parameter('rotate_image', False)

        # ==========================================================
        # Line detection parameters
        # ==========================================================
        self.declare_parameter('line_resize_width', 320)
        self.declare_parameter('line_resize_height', 240)
        self.declare_parameter('line_roi_top', 0.45)

        self.declare_parameter('blur_kernel', 7)
        self.declare_parameter('morph_kernel', 5)

        self.declare_parameter('min_area', 400)
        self.declare_parameter('max_area', 100000)
        self.declare_parameter('score_distance_weight', 8.0)

        self.declare_parameter('lookahead_row', 0.55) #Anterior 0.25
        self.declare_parameter('lost_timeout', 3.00)

        self.declare_parameter('trap_top_frac', 0.30) #anterior 0.10
        self.declare_parameter('trap_bottom_frac', 0.65)

        # Controls how high the triangular/trapezoid mask starts inside the
        # line ROI. IMPORTANT: this is applied after rh/rw exist.
        self.declare_parameter('trap_top_row_frac', 0.45)

        # ==========================================================
        # Traffic light parameters
        # ==========================================================
        self.declare_parameter('traffic_process_fps', 12.0)

        # Top-right ROI. Default: right-most upper zone.
        # X = 0.75 to 1.00, Y = 0.00 to 0.50.
        self.declare_parameter('traffic_roi_x_min_frac', 0.75)
        self.declare_parameter('traffic_roi_x_max_frac', 1.00)
        self.declare_parameter('traffic_roi_y_min_frac', 0.00)
        self.declare_parameter('traffic_roi_y_max_frac', 0.50)

        # Kept for compatibility with older launch files. In this version,
        # traffic_roi_y_max_frac is the parameter that defines the traffic ROI.
        self.declare_parameter('traffic_roi_bottom', 0.72)

        # Candidate validation in GLOBAL frame coordinates.
        self.declare_parameter('traffic_min_center_x_frac', 0.75)
        self.declare_parameter('traffic_max_center_x_frac', 1.00)
        self.declare_parameter('traffic_min_center_y_frac', 0.00)
        self.declare_parameter('traffic_max_center_y_frac', 0.52)

        # Minimum detection scores.
        self.declare_parameter('min_score_red', 55.0)
        self.declare_parameter('min_score_yellow', 28.0)
        self.declare_parameter('min_score_green', 45.0)

        # Brightness focus: rejects colored classroom objects that are not active LEDs.
        self.declare_parameter('traffic_bright_percentile', 92.0)
        self.declare_parameter('traffic_min_v_floor', 95.0)
        self.declare_parameter('traffic_min_candidate_mean_v', 105.0)
        self.declare_parameter('traffic_min_candidate_peak_v', 145.0)
        self.declare_parameter('traffic_min_candidate_bright_ratio', 0.20)

        # Blob filtering
        self.declare_parameter('traffic_min_area', 12.0)
        self.declare_parameter('traffic_max_area', 3500.0)
        self.declare_parameter('traffic_min_circularity', 0.12)
        self.declare_parameter('traffic_min_fill_ratio', 0.14)
        self.declare_parameter('traffic_max_aspect_ratio', 3.2)

        # Tracking/lost state for temporal logic.
        self.declare_parameter('roi_margin', 180)
        self.declare_parameter('max_lost_frames', 12)

        # Anti-flicker
        self.declare_parameter('yellow_hold_max', 16)
        self.declare_parameter('traffic_buffer_size', 10)
        self.declare_parameter('red_votes_required', 2)
        self.declare_parameter('yellow_votes_required', 2)
        self.declare_parameter('green_votes_required', 2)
        self.declare_parameter('unknown_votes_required', 7)

        # ==========================================================
        # Debug parameters
        # ==========================================================
        self.declare_parameter('debug', True)
        self.declare_parameter('debug_fps', 5.0)
        self.declare_parameter('debug_jpeg_quality', 45)

        self.debug_enabled = bool(self.get_parameter('debug').value)

        # ==========================================================
        # Publishers
        # ==========================================================
        self.error_pub = self.create_publisher(
            Float32MultiArray,
            '/perception/line_error',
            10
        )

        self.detected_pub = self.create_publisher(
            Bool,
            '/perception/line_detected',
            10
        )

        self.traffic_state_pub = self.create_publisher(
            String,
            '/traffic_light_state',
            10
        )

        self.traffic_action_pub = self.create_publisher(
            String,
            '/traffic_light_action',
            10
        )

        # Only create debug topics when debug is enabled. This prevents debug
        # topics from appearing in nodebug mode and avoids unnecessary overhead.
        self.line_debug_pub = None
        self.traffic_debug_pub = None

        if self.debug_enabled:
            self.line_debug_pub = self.create_publisher(
                CompressedImage,
                '/perception/debug/compressed',
                10
            )

            self.traffic_debug_pub = self.create_publisher(
                CompressedImage,
                '/traffic_debug/compressed',
                10
            )

        # ==========================================================
        # Subscriber
        # ==========================================================
        self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        # ==========================================================
        # Internal state: line
        # ==========================================================
        self.last_error_main = 0.0
        self.last_error_look = 0.0
        self.last_line_time = self.get_clock().now()

        self.frame_count = 0
        self.last_fps_time = self.get_clock().now()

        self.latest_line_debug = None

        # ==========================================================
        # Internal state: traffic
        # ==========================================================
        self.kernel_morph = np.ones((5, 5), np.uint8)

        self.tracking = False
        self.last_bbox = None
        self.lost_count = 0
        self.traffic_frame_count = 0

        buffer_size = int(self.get_parameter('traffic_buffer_size').value)
        self.state_buffer = deque(maxlen=buffer_size)

        self.final_state = 'UNKNOWN'
        self.yellow_hold_frames = 0

        self.latest_traffic_debug = None
        self.last_traffic_process_time = self.get_clock().now()
        self.current_traffic_vthr = 110.0

        # ==========================================================
        # Debug timing
        # ==========================================================
        self.last_debug_time = self.get_clock().now()
        self.last_traffic_log = ''

        self.get_logger().info(
            'PerceptionNode started: line detection + bright TOP-RIGHT traffic ROI'
        )

    # ==============================================================
    # Utility
    # ==============================================================

    def seconds_since(self, past_time):
        now = self.get_clock().now()
        return (now - past_time).nanoseconds / 1e9

    def publish_compressed(self, publisher, image_bgr):
        if publisher is None or image_bgr is None:
            return

        quality = int(self.get_parameter('debug_jpeg_quality').value)

        ret, buffer = cv2.imencode(
            '.jpg',
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        if not ret:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = buffer.tobytes()
        publisher.publish(msg)

    def clamp_frac(self, value, low=0.0, high=1.0):
        return max(low, min(high, float(value)))

    # ==============================================================
    # Line pipeline
    # ==============================================================

    def preprocess_line(self, frame):
        resize_w = int(self.get_parameter('line_resize_width').value)
        resize_h = int(self.get_parameter('line_resize_height').value)

        roi_top = float(self.get_parameter('line_roi_top').value)

        blur_k = int(self.get_parameter('blur_kernel').value)
        morph_k = int(self.get_parameter('morph_kernel').value)

        top_frac = float(self.get_parameter('trap_top_frac').value)
        bottom_frac = float(self.get_parameter('trap_bottom_frac').value)
        top_row_frac = float(self.get_parameter('trap_top_row_frac').value)

        if blur_k % 2 == 0:
            blur_k += 1

        if morph_k % 2 == 0:
            morph_k += 1

        frame_small = cv2.resize(frame, (resize_w, resize_h))
        h, w = frame_small.shape[:2]

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        roi_y = int(h * roi_top)
        roi_gray = gray[roi_y:h, :]

        rh, rw = roi_gray.shape[:2]

        # BUG FIX:
        # rh/rw must exist before using trap_top_row_frac.
        top_row_frac = float(np.clip(top_row_frac, 0.0, 0.95))
        top_row = int(rh * top_row_frac)

        blurred = cv2.GaussianBlur(roi_gray, (blur_k, blur_k), 0)

        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (morph_k, morph_k)
        )

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        top_w = int(rw * top_frac)
        bot_w = int(rw * bottom_frac)

        top_x1 = (rw - top_w) // 2
        top_x2 = top_x1 + top_w

        bot_x1 = (rw - bot_w) // 2
        bot_x2 = bot_x1 + bot_w

        polygon = np.array([
            [top_x1, top_row],
            [top_x2, top_row],
            [bot_x2, rh - 1],
            [bot_x1, rh - 1],
        ], dtype=np.int32)

        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        binary = cv2.bitwise_and(binary, binary, mask=mask)

        return binary, roi_gray, polygon, w

    def detect_line(self, binary, image_width):
        min_area = float(self.get_parameter('min_area').value)
        max_area = float(self.get_parameter('max_area').value)
        weight = float(self.get_parameter('score_distance_weight').value)

        center_x = image_width / 2.0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        best_label = -1
        best_score = -float('inf')

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if not (min_area <= area <= max_area):
                continue

            distance = abs(centroids[i][0] - center_x)
            score = area - weight * distance

            if score > best_score:
                best_score = score
                best_label = i

        return best_label, labels, centroids, stats

    def compute_line_errors(self, best_label, labels, centroids, binary_width):
        lookahead_row = float(self.get_parameter('lookahead_row').value)

        center = binary_width / 2.0
        roi_h = labels.shape[0]

        line_detected = best_label != -1

        cx_main = None
        cx_look = None
        cy_main = None

        if line_detected:
            row_main = int(centroids[best_label][1])
            row_main = int(np.clip(row_main, 0, roi_h - 1))
            cy_main = row_main

            cols_main = np.where(labels[row_main, :] == best_label)[0]
            if cols_main.size > 0:
                cx_main = float(cols_main[0] + cols_main[-1]) / 2.0

            row_look = int(roi_h * lookahead_row)
            row_look = int(np.clip(row_look, 0, roi_h - 1))

            cols_look = np.where(labels[row_look, :] == best_label)[0]
            if cols_look.size > 0:
                cx_look = float(cols_look[0] + cols_look[-1]) / 2.0

        error_main = (center - cx_main) / center if cx_main is not None else None
        error_look = (center - cx_look) / center if cx_look is not None else None

        return (
            line_detected,
            cx_main,
            cx_look,
            cy_main,
            error_main,
            error_look
        )

    def publish_line(self, line_detected, error_main, error_look):
        now = self.get_clock().now()
        lost_timeout = float(self.get_parameter('lost_timeout').value)

        if line_detected and error_main is not None:
            self.last_error_main = error_main
            self.last_error_look = error_look if error_look is not None else error_main
            self.last_line_time = now

            out_main = self.last_error_main
            out_look = self.last_error_look

        else:
            elapsed = (now - self.last_line_time).nanoseconds / 1e9

            if elapsed < lost_timeout:
                out_main = self.last_error_main
                out_look = self.last_error_look
            else:
                out_main = 0.0
                out_look = 0.0

        error_msg = Float32MultiArray()
        error_msg.data = [float(out_main), float(out_look)]
        self.error_pub.publish(error_msg)

        detected_msg = Bool()
        detected_msg.data = bool(line_detected)
        self.detected_pub.publish(detected_msg)

    def create_line_debug(
        self,
        roi_gray,
        polygon,
        best_label,
        stats,
        cx_main,
        cx_look,
        cy_main,
        image_width,
        line_detected,
        error_main,
        error_look
    ):
        vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        roi_h = vis.shape[0]

        cv2.polylines(vis, [polygon], True, (255, 255, 0), 2)

        cv2.line(
            vis,
            (image_width // 2, 0),
            (image_width // 2, roi_h),
            (0, 255, 0),
            1
        )

        lookahead_row = float(self.get_parameter('lookahead_row').value)
        lh_y = int(roi_h * lookahead_row)

        cv2.line(
            vis,
            (0, lh_y),
            (image_width, lh_y),
            (0, 128, 255),
            1
        )

        if best_label != -1:
            bx = stats[best_label, cv2.CC_STAT_LEFT]
            by = stats[best_label, cv2.CC_STAT_TOP]
            bw = stats[best_label, cv2.CC_STAT_WIDTH]
            bh = stats[best_label, cv2.CC_STAT_HEIGHT]

            cv2.rectangle(
                vis,
                (bx, by),
                (bx + bw, by + bh),
                (0, 255, 255),
                2
            )

            if cx_main is not None and cy_main is not None:
                cv2.circle(
                    vis,
                    (int(cx_main), int(cy_main)),
                    6,
                    (0, 0, 255),
                    -1
                )

                cv2.line(
                    vis,
                    (int(cx_main), int(cy_main)),
                    (image_width // 2, int(cy_main)),
                    (0, 0, 255),
                    2
                )

            if cx_look is not None:
                cv2.circle(
                    vis,
                    (int(cx_look), lh_y),
                    6,
                    (0, 128, 255),
                    -1
                )

        e_main_txt = 'None' if error_main is None else f'{error_main:.2f}'
        e_look_txt = 'None' if error_look is None else f'{error_look:.2f}'

        panel_text_1 = f'LINE:{line_detected}'
        panel_text_2 = f'M:{e_main_txt} L:{e_look_txt}'

        cv2.rectangle(vis, (4, 4), (150, 42), (0, 0, 0), -1)

        cv2.putText(
            vis,
            panel_text_1,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            vis,
            panel_text_2,
            (8, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

        return vis

    # ==============================================================
    # TOP-RIGHT traffic light pipeline
    # ==============================================================

    def clean_mask(self, mask):
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.erode(mask, self.kernel_morph, iterations=1)
        mask = cv2.dilate(mask, self.kernel_morph, iterations=2)
        return mask

    def get_traffic_roi_bounds(self, frame):
        h, w = frame.shape[:2]

        x_min_frac = self.clamp_frac(
            self.get_parameter('traffic_roi_x_min_frac').value
        )
        x_max_frac = self.clamp_frac(
            self.get_parameter('traffic_roi_x_max_frac').value
        )
        y_min_frac = self.clamp_frac(
            self.get_parameter('traffic_roi_y_min_frac').value
        )
        y_max_frac = self.clamp_frac(
            self.get_parameter('traffic_roi_y_max_frac').value
        )

        # Safety normalization in case launch sends inverted values.
        if x_max_frac <= x_min_frac:
            x_min_frac, x_max_frac = 0.75, 1.00

        if y_max_frac <= y_min_frac:
            y_min_frac, y_max_frac = 0.00, 0.50

        x1 = int(w * x_min_frac)
        x2 = int(w * x_max_frac)
        y1 = int(h * y_min_frac)
        y2 = int(h * y_max_frac)

        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, x1 + 1, w))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, y1 + 1, h))

        return x1, y1, x2, y2

    def get_dynamic_traffic_roi(self, frame):
        # Name kept for compatibility with older code.
        # This version returns ONLY the selected top-right traffic ROI.
        self.traffic_frame_count += 1

        x1, y1, x2, y2 = self.get_traffic_roi_bounds(frame)
        roi = frame[y1:y2, x1:x2]

        return roi, x1, y1, True

    def make_color_masks(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        v = hsv[:, :, 2]

        # Adaptive brightness floor calculated ONLY inside traffic ROI.
        # Percentile focuses detection on bright LED-like pixels.
        v_mean = float(np.mean(v))
        v_std = float(np.std(v))
        bright_percentile = float(self.get_parameter('traffic_bright_percentile').value)
        min_v_floor = float(self.get_parameter('traffic_min_v_floor').value)
        v_perc = float(np.percentile(v, bright_percentile))

        adaptive_v = int(np.clip(
            max(min_v_floor, v_mean + 0.65 * v_std, v_perc - 4.0),
            70,
            210
        ))

        red_mask_1 = cv2.inRange(
            hsv,
            np.array([0, 50, 70]),
            np.array([14, 255, 255])
        )

        red_mask_2 = cv2.inRange(
            hsv,
            np.array([166, 50, 70]),
            np.array([179, 255, 255])
        )

        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        yellow_mask = cv2.inRange(
            hsv,
            np.array([10, 32, 70]),
            np.array([47, 255, 255])
        )

        green_mask = cv2.inRange(
            hsv,
            np.array([38, 32, 65]),
            np.array([108, 255, 255])
        )

        # Main classroom-noise rejection step:
        # color must also be bright enough to look like an active LED.
        bright_mask = cv2.inRange(
            hsv,
            np.array([0, 0, adaptive_v]),
            np.array([179, 255, 255])
        )

        red_mask = cv2.bitwise_and(red_mask, bright_mask)
        yellow_mask = cv2.bitwise_and(yellow_mask, bright_mask)
        green_mask = cv2.bitwise_and(green_mask, bright_mask)

        red_mask = self.clean_mask(red_mask)
        yellow_mask = self.clean_mask(yellow_mask)
        green_mask = self.clean_mask(green_mask)

        self.current_traffic_vthr = adaptive_v

        return hsv, red_mask, yellow_mask, green_mask, adaptive_v

    def evaluate_blob(self, contour, mask, hsv, label, offset_x, offset_y, frame_h, frame_w):
        area = cv2.contourArea(contour)

        min_area = float(self.get_parameter('traffic_min_area').value)
        max_area = float(self.get_parameter('traffic_max_area').value)

        if area < min_area or area > max_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)

        if w < 3 or h < 3:
            return None

        aspect = max(w / float(h), h / float(w))
        max_aspect = float(self.get_parameter('traffic_max_aspect_ratio').value)

        if aspect > max_aspect:
            return None

        rect_area = float(w * h)
        fill_ratio = area / rect_area if rect_area > 0 else 0.0

        min_fill = float(self.get_parameter('traffic_min_fill_ratio').value)
        if fill_ratio < min_fill:
            return None

        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0

        if perimeter > 0:
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

        min_circularity = float(self.get_parameter('traffic_min_circularity').value)
        if circularity < min_circularity:
            return None

        cx = x + w / 2.0
        cy = y + h / 2.0

        global_cx = cx + offset_x
        global_cy = cy + offset_y

        min_y_frac = float(self.get_parameter('traffic_min_center_y_frac').value)
        max_y_frac = float(self.get_parameter('traffic_max_center_y_frac').value)
        min_x_frac = float(self.get_parameter('traffic_min_center_x_frac').value)
        max_x_frac = float(self.get_parameter('traffic_max_center_x_frac').value)

        if global_cy < frame_h * min_y_frac:
            return None

        if global_cy > frame_h * max_y_frac:
            return None

        if global_cx < frame_w * min_x_frac:
            return None

        if global_cx > frame_w * max_x_frac:
            return None

        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [contour], -1, 255, -1)

        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        mean_v = cv2.mean(v_channel, mask=blob_mask)[0]
        mean_s = cv2.mean(s_channel, mask=blob_mask)[0]

        blob_pixels = v_channel[blob_mask > 0]
        if blob_pixels.size == 0:
            return None

        peak_v = float(np.max(blob_pixels))
        vthr = float(getattr(self, 'current_traffic_vthr', 110.0))
        bright_ratio = float(np.count_nonzero(blob_pixels >= vthr)) / float(blob_pixels.size)

        min_mean_v = float(self.get_parameter('traffic_min_candidate_mean_v').value)
        min_peak_v = float(self.get_parameter('traffic_min_candidate_peak_v').value)
        min_bright_ratio = float(self.get_parameter('traffic_min_candidate_bright_ratio').value)

        # Reject colored but non-bright objects such as tables, bags, posters, etc.
        if mean_v < min_mean_v and peak_v < min_peak_v:
            return None

        if bright_ratio < min_bright_ratio:
            return None

        score = (
            np.sqrt(area)
            * (mean_v / 255.0) ** 2.2
            * (peak_v / 255.0) ** 1.2
            * (0.35 + 0.65 * (mean_s / 255.0))
            * (0.55 + 0.45 * circularity)
            * (0.60 + 0.40 * fill_ratio)
            * (0.55 + 0.45 * bright_ratio)
            * 100.0
        )

        return {
            'label': label,
            'bbox': (x, y, w, h),
            'global_bbox': (x + offset_x, y + offset_y, w, h),
            'area': area,
            'mean_v': mean_v,
            'mean_s': mean_s,
            'peak_v': peak_v,
            'bright_ratio': bright_ratio,
            'circularity': circularity,
            'fill_ratio': fill_ratio,
            'score': score,
            'center': (global_cx, global_cy),
        }

    def best_color_candidate(self, mask, hsv, label, offset_x, offset_y, frame_h, frame_w):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best = None
        all_candidates = []

        for contour in contours:
            candidate = self.evaluate_blob(
                contour=contour,
                mask=mask,
                hsv=hsv,
                label=label,
                offset_x=offset_x,
                offset_y=offset_y,
                frame_h=frame_h,
                frame_w=frame_w
            )

            if candidate is None:
                continue

            all_candidates.append(candidate)

            if best is None or candidate['score'] > best['score']:
                best = candidate

        return best, all_candidates

    def choose_traffic_detection(self, red_best, yellow_best, green_best):
        min_r = float(self.get_parameter('min_score_red').value)
        min_y = float(self.get_parameter('min_score_yellow').value)
        min_g = float(self.get_parameter('min_score_green').value)

        candidates = []

        if red_best is not None and red_best['score'] >= min_r:
            candidates.append(red_best)

        if yellow_best is not None and yellow_best['score'] >= min_y:
            candidates.append(yellow_best)

        if green_best is not None and green_best['score'] >= min_g:
            candidates.append(green_best)

        if not candidates:
            return 'UNKNOWN', None

        best = max(candidates, key=lambda c: c['score'])
        return best['label'], best

    def update_traffic_temporal_filter(self, detected):
        yellow_hold_max = int(self.get_parameter('yellow_hold_max').value)

        if detected == 'YELLOW':
            self.yellow_hold_frames = yellow_hold_max

        elif self.yellow_hold_frames > 0:
            self.yellow_hold_frames -= 1

        if self.yellow_hold_frames > 0 and detected == 'UNKNOWN':
            detected_for_buffer = 'YELLOW'
        else:
            detected_for_buffer = detected

        self.state_buffer.append(detected_for_buffer)

        red_count = self.state_buffer.count('RED')
        yellow_count = self.state_buffer.count('YELLOW')
        green_count = self.state_buffer.count('GREEN')
        unknown_count = self.state_buffer.count('UNKNOWN')

        red_votes = int(self.get_parameter('red_votes_required').value)
        yellow_votes = int(self.get_parameter('yellow_votes_required').value)
        green_votes = int(self.get_parameter('green_votes_required').value)
        unknown_votes = int(self.get_parameter('unknown_votes_required').value)

        if red_count >= red_votes:
            self.final_state = 'RED'
        elif yellow_count >= yellow_votes:
            self.final_state = 'YELLOW'
        elif green_count >= green_votes:
            self.final_state = 'GREEN'
        elif unknown_count >= unknown_votes:
            self.final_state = 'UNKNOWN'

        return (
            self.final_state,
            red_count,
            yellow_count,
            green_count,
            unknown_count
        )

    def state_to_action(self, state):
        if state == 'RED':
            return 'DETENIDO'
        if state == 'YELLOW':
            return 'BAJANDO VELOCIDAD'
        if state == 'GREEN':
            return 'AVANZANDO'
        return 'BUSCANDO'

    def color_for_state(self, state):
        if state == 'RED':
            return (0, 0, 255)
        if state == 'YELLOW':
            return (0, 255, 255)
        if state == 'GREEN':
            return (0, 255, 0)
        return (180, 180, 180)

    def process_traffic(self, frame):
        roi, offset_x, offset_y, full_mode = self.get_dynamic_traffic_roi(frame)

        frame_h, frame_w = frame.shape[:2]

        hsv, red_mask, yellow_mask, green_mask, adaptive_v = self.make_color_masks(roi)

        red_best, red_candidates = self.best_color_candidate(
            red_mask,
            hsv,
            'RED',
            offset_x,
            offset_y,
            frame_h,
            frame_w
        )

        yellow_best, yellow_candidates = self.best_color_candidate(
            yellow_mask,
            hsv,
            'YELLOW',
            offset_x,
            offset_y,
            frame_h,
            frame_w
        )

        green_best, green_candidates = self.best_color_candidate(
            green_mask,
            hsv,
            'GREEN',
            offset_x,
            offset_y,
            frame_h,
            frame_w
        )

        detected, best_candidate = self.choose_traffic_detection(
            red_best,
            yellow_best,
            green_best
        )

        if detected != 'UNKNOWN' and best_candidate is not None:
            gx, gy, gw, gh = best_candidate['global_bbox']

            self.last_bbox = (gx, gy, gw, gh)
            self.tracking = True
            self.lost_count = 0

        else:
            self.lost_count += 1

            max_lost = int(self.get_parameter('max_lost_frames').value)

            if self.lost_count > max_lost:
                self.tracking = False
                self.last_bbox = None
                self.state_buffer.clear()
                self.final_state = 'UNKNOWN'
                self.yellow_hold_frames = 0

        (
            stable_state,
            red_count,
            yellow_count,
            green_count,
            unknown_count
        ) = self.update_traffic_temporal_filter(detected)

        action = self.state_to_action(stable_state)

        state_msg = String()
        state_msg.data = stable_state
        self.traffic_state_pub.publish(state_msg)

        action_msg = String()
        action_msg.data = action
        self.traffic_action_pub.publish(action_msg)

        self.latest_traffic_debug = {
            'full_mode': full_mode,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'detected': detected,
            'stable_state': stable_state,
            'action': action,
            'best_candidate': best_candidate,
            'red_best': red_best,
            'yellow_best': yellow_best,
            'green_best': green_best,
            'red_candidates': red_candidates,
            'yellow_candidates': yellow_candidates,
            'green_candidates': green_candidates,
            'counts': (red_count, yellow_count, green_count, unknown_count),
            'adaptive_v': adaptive_v,
        }

        r_score = 0 if red_best is None else int(red_best['score'])
        y_score = 0 if yellow_best is None else int(yellow_best['score'])
        g_score = 0 if green_best is None else int(green_best['score'])

        log = (
            f'TRAFFIC RIGHT_ROI raw={detected} stable={stable_state} action={action} | '
            f'R={r_score} Y={y_score} G={g_score} | '
            f'tracking={self.tracking} lost={self.lost_count} Vthr={adaptive_v}'
        )

        if log != self.last_traffic_log:
            self.get_logger().info(log)
            self.last_traffic_log = log

    def draw_candidate(self, vis, candidate, color, thickness=1):
        if candidate is None:
            return

        gx, gy, gw, gh = candidate['global_bbox']
        score = int(candidate['score'])

        cv2.rectangle(
            vis,
            (int(gx), int(gy)),
            (int(gx + gw), int(gy + gh)),
            color,
            thickness
        )

        cv2.putText(
            vis,
            f'{candidate["label"]}:{score}',
            (int(gx), max(20, int(gy) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA
        )

    def create_traffic_debug(self, frame):
        vis = frame.copy()
        h, w = vis.shape[:2]

        x1, y1, x2, y2 = self.get_traffic_roi_bounds(frame)

        # Make selected ROI obvious: darken everything else.
        dark = (vis * 0.35).astype(np.uint8)
        dark[y1:y2, x1:x2] = vis[y1:y2, x1:x2]
        vis = dark

        # Draw 4 quadrants.
        mid_x = w // 2
        mid_y = h // 2
        cv2.line(vis, (mid_x, 0), (mid_x, h), (255, 255, 255), 1)
        cv2.line(vis, (0, mid_y), (w, mid_y), (255, 255, 255), 1)

        # Highlight traffic ROI.
        cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 3)

        cv2.putText(
            vis,
            'ACTIVE TRAFFIC ROI: RIGHT-TOP ZONE',
            (x1 + 8, y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            vis,
            f'x:{x1}-{x2} y:{y1}-{y2}',
            (x1 + 8, y1 + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        if self.latest_traffic_debug is None:
            return cv2.resize(vis, (640, 480))

        data = self.latest_traffic_debug

        stable_state = data['stable_state']
        action = data['action']
        detected = data['detected']

        red_best = data['red_best']
        yellow_best = data['yellow_best']
        green_best = data['green_best']
        best_candidate = data['best_candidate']

        red_count, yellow_count, green_count, unknown_count = data['counts']
        adaptive_v = data['adaptive_v']

        self.draw_candidate(vis, red_best, (0, 0, 255), 1)
        self.draw_candidate(vis, yellow_best, (0, 255, 255), 1)
        self.draw_candidate(vis, green_best, (0, 255, 0), 1)

        if best_candidate is not None:
            self.draw_candidate(
                vis,
                best_candidate,
                self.color_for_state(best_candidate['label']),
                3
            )

        r_score = 0 if red_best is None else int(red_best['score'])
        y_score = 0 if yellow_best is None else int(yellow_best['score'])
        g_score = 0 if green_best is None else int(green_best['score'])

        panel_color = self.color_for_state(stable_state)

        cv2.rectangle(vis, (8, 42), (335, 164), (0, 0, 0), -1)
        cv2.rectangle(vis, (8, 42), (335, 164), panel_color, 2)

        lines = [
            f'{stable_state} | {action}',
            f'raw:{detected} Vthr:{adaptive_v}',
            f'Score R:{r_score} Y:{y_score} G:{g_score}',
            f'Buf R:{red_count} Y:{yellow_count} G:{green_count} U:{unknown_count}',
            f'ROI x:{x1}-{x2} y:{y1}-{y2}',
        ]

        y0 = 62

        for i, text in enumerate(lines):
            color = panel_color if i == 0 else (255, 255, 255)

            cv2.putText(
                vis,
                text,
                (16, y0 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA
            )

        return cv2.resize(vis, (640, 480))

    # ==============================================================
    # Main callback
    # ==============================================================

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as error:
            self.get_logger().error(f'cv_bridge error: {error}')
            return

        if bool(self.get_parameter('rotate_image').value):
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Line processing
        binary, roi_gray, polygon, line_width = self.preprocess_line(frame)

        best_label, labels, centroids, stats = self.detect_line(
            binary,
            line_width
        )

        (
            line_detected,
            cx_main,
            cx_look,
            cy_main,
            error_main,
            error_look
        ) = self.compute_line_errors(
            best_label,
            labels,
            centroids,
            line_width
        )

        self.publish_line(line_detected, error_main, error_look)

        if self.debug_enabled:
            self.latest_line_debug = self.create_line_debug(
                roi_gray=roi_gray,
                polygon=polygon,
                best_label=best_label,
                stats=stats,
                cx_main=cx_main,
                cx_look=cx_look,
                cy_main=cy_main,
                image_width=line_width,
                line_detected=line_detected,
                error_main=error_main,
                error_look=error_look
            )

        # Traffic processing throttled
        traffic_fps = float(self.get_parameter('traffic_process_fps').value)
        traffic_period = 1.0 / traffic_fps if traffic_fps > 0 else 0.1

        if self.seconds_since(self.last_traffic_process_time) >= traffic_period:
            self.last_traffic_process_time = self.get_clock().now()
            self.process_traffic(frame)

        # Debug processing throttled
        debug_fps = float(self.get_parameter('debug_fps').value)
        debug_period = 1.0 / debug_fps if debug_fps > 0 else 0.2

        if self.debug_enabled and self.seconds_since(self.last_debug_time) >= debug_period:
            self.last_debug_time = self.get_clock().now()

            if self.latest_line_debug is not None:
                self.publish_compressed(
                    self.line_debug_pub,
                    self.latest_line_debug
                )

            traffic_debug = self.create_traffic_debug(frame)
            self.publish_compressed(
                self.traffic_debug_pub,
                traffic_debug
            )

        # Logs
        self.frame_count += 1

        if self.frame_count % 30 == 0:
            now = self.get_clock().now()
            dt = (now - self.last_fps_time).nanoseconds / 1e9
            self.last_fps_time = now

            fps_real = 30.0 / dt if dt > 0 else 0.0

            e_main_txt = 'None' if error_main is None else f'{error_main:.3f}'
            e_look_txt = 'None' if error_look is None else f'{error_look:.3f}'

            self.get_logger().info(
                f'LINE fps={fps_real:.1f} detected={line_detected} '
                f'e_main={e_main_txt} e_look={e_look_txt} | '
                f'TRAFFIC={self.final_state}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
