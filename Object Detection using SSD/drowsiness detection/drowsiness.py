import cv2
import dlib
import numpy as np
from imutils import face_utils
from pygame import mixer

# 1. Initialize alarm, thresholds, counters, etc.
THRES_EYES = 6        # Eye “closed” threshold
DROWSY_LIST_SIZE = 10 # Number of frames to check for drowsiness
FACING_LIST_SIZE = 120# ~4 seconds at ~30 fps (adjust based on your FPS)
YAW_ANGLE_THRESH = 20 # If abs(yaw) > 20 degrees => not facing forward

mixer.init()
sound = mixer.Sound('alarm.wav')

dlist = []            # For drowsiness (eyes-closed) detection
facing_list = []      # For not-facing-forward detection

# 2. Dlib setup for facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# 3. 3D model points for solvePnP
model_3d_points = np.array([
    (0.0,      0.0,      0.0   ),  # Nose tip (landmark 30)
    (0.0,     -330.0,    -65.0 ),  # Chin (landmark 8)
    (-225.0,   170.0,    -135.0),  # Left eye left corner (landmark 36)
    (225.0,    170.0,    -135.0),  # Right eye right corner (landmark 45)
    (-150.0,  -150.0,    -125.0),  # Left Mouth corner (landmark 48)
    (150.0,   -150.0,    -125.0)   # Right mouth corner (landmark 54)
], dtype=np.float64)

# (Optional) connectivity list for drawing a "mask"
FACE_CONNECTIONS = [
    # Jaw line
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),
    (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),
    # Left eyebrow
    (17,18),(18,19),(19,20),(20,21),
    # Right eyebrow
    (22,23),(23,24),(24,25),(25,26),
    # Nose bridge
    (27,28),(28,29),(29,30),
    # Nose base
    (31,32),(32,33),(33,34),(34,35),
    # Left eye
    (36,37),(37,38),(38,39),(39,40),(40,41),(41,36),
    # Right eye
    (42,43),(43,44),(44,45),(45,46),(46,47),(47,42),
    # Outer lip
    (48,49),(49,50),(50,51),(51,52),(52,53),(53,54),(54,55),(55,56),
    (56,57),(57,58),(58,59),(59,48),
    # Inner lip
    (60,61),(61,62),(62,63),(63,64),(64,65),(65,66),(66,67),(67,60)
]

def dist(a, b):
    """
    Simple Euclidean distance function.
    """
    x1, y1 = a
    x2, y2 = b
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def get_head_pose(shape, size):
    """
    Use solvePnP to estimate head pose given 6 specific landmark points.
    shape: (68 x 2) array of (x, y)
    size:  (height, width) of the frame
    Returns: (yaw, pitch, roll) angles in degrees.
    """
    print("[DEBUG] get_head_pose() called.")

    # Create a list of 6 key points: nose tip, chin, eye corners, mouth corners.
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]   # Right mouth corner
    ], dtype=np.float64)

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1       ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Solve for pose
    success, rotation_vector, _ = cv2.solvePnP(
        model_3d_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    print(f"[DEBUG] solvePnP success: {success}")

    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)

    if sy < 1e-6:
        yaw   = np.degrees(np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1]))
        pitch = np.degrees(np.arctan2(-rotation_matrix[2,0], sy))
        roll  = 0
    else:
        yaw   = np.degrees(np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0]))
        pitch = np.degrees(np.arctan2(-rotation_matrix[2,0], sy))
        roll  = np.degrees(np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]))

    print(f"[DEBUG] yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
    return yaw, pitch, roll

print("[DEBUG] Starting main loop...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[DEBUG] Frame not received. Exiting...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    print(f"[DEBUG] Number of faces detected: {len(rects)}")

    if len(rects) == 0:
        print("[DEBUG] No faces detected. Stopping alarm if playing.")
        try:
            sound.stop()
        except:
            pass
        cv2.imshow("Output", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            print("[DEBUG] ESC pressed. Exiting...")
            break
        continue

    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Draw small circles for each landmark
    for (x, y) in shape:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Connect the landmarks (mask effect)
    for (start, end) in FACE_CONNECTIONS:
        x1, y1 = shape[start]
        x2, y2 = shape[end]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # ========== EYE-BASED DROWSINESS DETECTION ========== 
    le_38 = shape[37]
    le_39 = shape[38]
    le_41 = shape[40]
    le_42 = shape[41]

    re_44 = shape[43]
    re_45 = shape[44]
    re_47 = shape[46]
    re_48 = shape[47]

    eye_metric = (
        dist(le_38, le_42) + 
        dist(le_39, le_41) + 
        dist(re_44, re_48) + 
        dist(re_45, re_47)
    ) / 4.0
    
    print(f"[DEBUG] eye_metric = {eye_metric:.2f}")

    dlist.append(eye_metric < THRES_EYES)
    if len(dlist) > DROWSY_LIST_SIZE:
        dlist.pop(0)

    drowsy_detected = (sum(dlist) >= 4)
    if drowsy_detected:
        print("[DEBUG] Drowsiness condition met (eyes closed in multiple frames).")

    # ========== HEAD ORIENTATION DETECTION ========== 
    h, w = frame.shape[:2]
    yaw, pitch, roll = get_head_pose(shape, (h, w))

    # Determine left/right/center based on yaw
    if yaw > YAW_ANGLE_THRESH:
        print("[DEBUG] Head is turned LEFT.")
    elif yaw < -YAW_ANGLE_THRESH:
        print("[DEBUG] Head is turned RIGHT.")
    else:
        print("[DEBUG] Head is roughly CENTERED.")

    not_facing_forward = (abs(yaw) > YAW_ANGLE_THRESH)
    facing_list.append(not_facing_forward)
    if len(facing_list) > FACING_LIST_SIZE:
        facing_list.pop(0)

    facing_away_detected = (sum(facing_list) > (FACING_LIST_SIZE // 2))
    if facing_away_detected:
        print("[DEBUG] Facing away condition met (yaw beyond threshold in recent frames).")

    # ========== ALARM TRIGGER ========== 
    if drowsy_detected or facing_away_detected:
        print("[DEBUG] ALARM condition triggered!")
        try:
            sound.play()
        except:
            pass
    else:
        print("[DEBUG] No alarm condition.")
        try:
            sound.stop()
        except:
            pass

    cv2.imshow("Output", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        print("[DEBUG] ESC pressed. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("[DEBUG] Program ended.")
