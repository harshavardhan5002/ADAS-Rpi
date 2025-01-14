import cv2
import numpy as np
import imutils
import math
from imutils.video import FPS

# -------------------------------------------------------
# CONFIGURATIONS
# -------------------------------------------------------
use_gpu = True
live_video = False
confidence_level = 0.3

# Collision threshold (in meters) for naive collision alert
COLLISION_THRESHOLD = 10.0

# Reference values for naive distance calculation
REF_DISTANCE = 10.0   # meters (arbitrary reference)
REF_WIDTH_PX = 100.0  # pixels (arbitrary reference width)

# -------------------------------------------------------
# LANE DETECTION UTILS
# -------------------------------------------------------
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=(0, 255, 0), thickness=2):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold,
                           np.array([]),
                           minLineLength=min_line_len,
                           maxLineGap=max_line_gap)

def calculate_slope_and_angle(line):
    x1, y1, x2, y2 = line
    # Add tiny offset to avoid division by zero
    slope = (y2 - y1) / (x2 - x1 + 1e-5)
    angle = math.degrees(math.atan(slope))
    return slope, angle

def process_lane(frame):
    """
    Detect and draw lane lines on the given frame.
    Also prints whether the road is straight or curving.
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # 4) Define region of interest
    imshape = frame.shape
    vertices = np.array([[
        (0, imshape[0]),
        (imshape[1] // 2, imshape[0] // 2 + 50),
        (imshape[1] // 2, imshape[0] // 2 + 50),
        (imshape[1], imshape[0])
    ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # 5) Hough Transform parameters
    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_len = 40
    max_line_gap = 20

    # 6) Detect lines
    lines = hough_lines(masked_edges, rho, theta, threshold,
                        min_line_len, max_line_gap)

    # 7) Determine if road is straight or curved
    if lines is not None:
        slopes, angles = zip(*[calculate_slope_and_angle(line[0]) for line in lines])
        avg_slope = np.mean(slopes)
        avg_angle = np.mean(angles)

        if np.abs(avg_slope) > 0.1:
            if avg_slope > 0:
                print(f"Curving to the RIGHT. Angle: {avg_angle:.2f}°")
            else:
                print(f"Curving to the LEFT. Angle: {avg_angle:.2f}°")
        else:
            print("The road is STRAIGHT")

    # 8) Draw the lines on a blank image
    lines_image = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)
    draw_lines(lines_image, lines)

    # 9) Overlay lane lines on the original frame
    result = cv2.addWeighted(frame, 0.8, lines_image, 1, 0)
    return result

# -------------------------------------------------------
# DISTANCE ESTIMATION (NAIVE)
# -------------------------------------------------------
def estimate_distance(box_width_px):
    """
    distance ~ (REF_DISTANCE * REF_WIDTH_PX) / box_width_px
    """
    if box_width_px <= 0:
        return None
    return (REF_DISTANCE * REF_WIDTH_PX) / box_width_px

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    fps = FPS().start()

    # Classes supported by the MobileNet SSD
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    # Random color array for each class (we will override some)
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    LIGHT_GREEN = (144, 238, 144)
    RED = (0, 0, 255)

    print("[INFO] Loading MobileNet SSD model...")
    net = cv2.dnn.readNetFromCaffe(
        'ssd_files/MobileNetSSD_deploy.prototxt',
        'ssd_files/MobileNetSSD_deploy.caffemodel'
    )

    # Optionally use GPU
    if use_gpu:
        print("[INFO] Setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    print("[INFO] Accessing video stream...")
    if live_video:
        vs = cv2.VideoCapture(0)
    else:
        vs = cv2.VideoCapture('test1.mp4')  # Update with your file path

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # ---------------------
        # 1) Object Detection
        # ---------------------
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        collision_risk = False  # Will become True if any car/person < threshold

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < confidence_level:
                continue

            idx = int(detections[0, 0, i, 1])
            label_name = CLASSES[idx]

            # Compute bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clamp to frame edges
            startX = max(0, startX)
            startY = max(0, startY)
            endX   = min(w, endX)
            endY   = min(h, endY)

            # Default bounding box color = light green
            box_color = LIGHT_GREEN
            label_text = f"{label_name}: {confidence*100:.2f}%"

            # Collision check for car/person
            if label_name in ["car", "person"]:
                box_width = endX - startX
                distance = estimate_distance(box_width)
                if distance is not None:
                    label_text += f" | Dist: {distance:.2f}m"
                    if distance < COLLISION_THRESHOLD:
                        collision_risk = True
                        box_color = RED  # Mark in RED

            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)

            # Draw label
            label_y = startY - 15 if (startY - 15) > 15 else startY + 15
            cv2.putText(frame, label_text, (startX, label_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 2)

        # If ANY detection is under threshold, show top-center warning text
        if collision_risk:
            warning_text = "COLLISION WARNING"
            font_scale = 1.5
            thickness = 3
            text_size, _ = cv2.getTextSize(warning_text,
                                           cv2.FONT_HERSHEY_DUPLEX,
                                           font_scale, thickness)
            text_width, text_height = text_size
            x_coord = int((w - text_width) / 2)
            y_coord = 60
            cv2.putText(frame,
                        warning_text,
                        (x_coord, y_coord),
                        cv2.FONT_HERSHEY_DUPLEX,
                        font_scale,
                        RED,
                        thickness)

        # ---------------------
        # 2) Lane Detection
        # ---------------------
        # We pass the frame (with bounding boxes) to lane detection.
        # Lane detection draws lines on top of the bounding boxes.
        frame_with_lanes = process_lane(frame)

        cv2.imshow('Object & Lane Detection', frame_with_lanes)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

        fps.update()

    fps.stop()
    vs.release()
    cv2.destroyAllWindows()

    print("[INFO] Elapsed time: {:.2f} seconds".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

if __name__ == "__main__":
    main()
