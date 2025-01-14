import cv2
import numpy as np
import imutils
from imutils.video import FPS

# -------------------------------------------------------
# CONFIGURATIONS
# -------------------------------------------------------
use_gpu = True
live_video = False
confidence_level = 0.3

# Use 10 meters as the threshold for collision
COLLISION_THRESHOLD = 10.0  

# Reference values for naive distance calculation
REF_DISTANCE = 10.0   # meters (arbitrary reference)
REF_WIDTH_PX = 100.0  # pixels (arbitrary reference width)

# -------------------------------------------------------
# DISTANCE ESTIMATION (NAIVE)
# -------------------------------------------------------
def estimate_distance(box_width_px):
    """
    Naive distance estimation from bounding-box width:
      distance ~ (REF_DISTANCE * REF_WIDTH_PX) / box_width_px
    """
    if box_width_px <= 0:
        return None
    distance = (REF_DISTANCE * REF_WIDTH_PX) / box_width_px
    return distance

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
fps = FPS().start()

# Classes supported by the MobileNet SSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Generate random colors for each class, but we will override with light green
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Use a light green color by default (BGR)
LIGHT_GREEN = (144, 238, 144)
RED = (0, 0, 255)

print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(
    'ssd_files/MobileNetSSD_deploy.prototxt',
    'ssd_files/MobileNetSSD_deploy.caffemodel'
)

# Optionally use GPU
if use_gpu:
    print("[INFO] Setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Video source
print("[INFO] Accessing video stream...")
if live_video:
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture('test1.mp4')

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    collision_risk = False  # Track if ANY object is under threshold

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidence_level:
            continue

        # Get class ID and bounding box
        idx = int(detections[0, 0, i, 1])
        label_name = CLASSES[idx]

        # Scale bounding box coords
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Clamp to frame boundaries
        startX = max(0, startX)
        startY = max(0, startY)
        endX   = min(w, endX)
        endY   = min(h, endY)

        # Default color is light green
        box_color = LIGHT_GREEN

        # Construct base label
        label_text = f"{label_name}: {confidence*100:.2f}%"

        # Only car or person triggers distance check
        if label_name in ["car", "person"]:
            box_width = endX - startX
            distance = estimate_distance(box_width)
            if distance is not None:
                label_text += f" | Dist: {distance:.2f}m"

                # If object is within the threshold
                if distance < COLLISION_THRESHOLD:
                    collision_risk = True
                    box_color = RED

        # Draw bounding box
        cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)

        # Place label above or below box
        label_y = startY - 15 if (startY - 15) > 15 else startY + 15
        cv2.putText(frame, label_text, (startX, label_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 2)

    # If ANY detection triggered collision
    if collision_risk:
        # Put big, bold text on top (centered horizontally, for example)
        warning_text = "COLLISION WARNING"
        font_scale = 1.5  # Increase as needed
        thickness = 3
        # Compute text size to help center
        text_size, _ = cv2.getTextSize(warning_text,
                                       cv2.FONT_HERSHEY_DUPLEX,
                                       font_scale, thickness)
        text_width, text_height = text_size

        # Coordinates for center alignment
        x_coord = int((w - text_width) / 2)
        y_coord = 60  # Just some top margin

        # Draw the text in red, bold
        cv2.putText(frame,
                    warning_text,
                    (x_coord, y_coord),
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_scale,
                    RED,
                    thickness)

    cv2.imshow('Object Detection & Collision Warning', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

    fps.update()

fps.stop()
vs.release()
cv2.destroyAllWindows()

print("[INFO] Elapsed time: {:.2f} seconds".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
