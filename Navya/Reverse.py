import time
import random

# Simulate hardware components (placeholders for actual hardware control)
def moveForward():
    print("Moving Forward")

def moveBack():
    print("Moving Backward")

def turnLeft():
    print("Turning Left")

def turnRight():
    print("Turning Right")

def stopMotors():
    print("Motors Stopped")

def lanecenter():
    print("Centering Lane")

def adaptiveCruiseControl():
    print("Adaptive Cruise Control")

# Global variable to track brake application time
last_brake_time = 0

def Reverse(backleft, backright, back, man_flag, direction):
    global last_brake_time
    current_time = time.time()
    # Check if brakes were applied recently
    if last_brake_time and (current_time - last_brake_time < 2):
        if direction in ['Backward', 'Left', 'Right']:
            print(f"Movement {direction} is temporarily disabled due to recent braking.")
            return

    if man_flag == 1:
        # Manual mode
        if direction == 'Forward':
            moveForward()
        elif direction == 'Backward':
            moveBack()
        elif direction == 'Left':
            turnLeft()
        elif direction == 'Right':
            turnRight()

    # Automatic mode
    # Example constants for distances
    dist1 = 50
    dist2 = 30
    dist3 = 20
    dist4 = 10
    distl = 15
    distr = 15

    brakes_applied = False
    if back > dist1:
        print("Lead Clear")
    elif dist1 >= back > dist2:
        print("Front1")
    elif dist2 >= back > dist3:
        print("Front2")
        brakes_applied = True
    elif back <= dist3:
        print("Front3")
        brakes_applied = True

        # Back Left
    if backleft > dist1:
        print("Back Left Ext")
    elif dist1 >= backleft > dist2:
        print("Back Left")
    elif dist2 >= backleft > dist3:
        print("Back Left - Brakes Applied")
        brakes_applied = True
    elif backleft <= dist4:
        print("Back Left - Steering Adjusted")
        brakes_applied = True

        # Back Right
    if backright > dist1:
        print("Back Right Ext")
    elif dist1 >= backright > dist2:
        print("Back Right")
    elif dist2 >= backright > dist3:
        print("Back Right - Brakes Applied")
        brakes_applied = True
    elif backright <= dist4:
        print("Back Right - Steering Adjusted")
        brakes_applied = True

    if brakes_applied:
        stopMotors()
        last_brake_time = current_time
