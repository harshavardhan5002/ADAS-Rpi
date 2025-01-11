import Drive
import Reverse
import Park
import Cruise
import random
import time
import threading
import tkinter as tk
import sys
import keyboard

# Global variables for distances, driving modes, speed, and thread control
distances = {'front': 0, 'back': 0, 'frontleft': 0, 'frontright': 0, 'backleft': 0, 'backright': 0}
modes = {'Drive': False, 'Reverse': False, 'Cruise': False, 'Park': True}
direction = None
speed = 0
threads_started = False
stop_threads = threading.Event()

# Debug mode flag
debug_mode = 1

# Create Tkinter windows
root = tk.Tk()
root.title("ADAS Sensor Monitor")
root.geometry("400x400")

# Sensor data canvas
canvas = tk.Canvas(root, width=400, height=300)
canvas.pack()

# Gear display label
gear_label = tk.Label(root, text="Current Gear: Park")
gear_label.pack()

# Battery percentage label
battery_label = tk.Label(root, text="Battery: N/A")
battery_label.pack()

# Speed label
speed_label = tk.Label(root, text="Speed: 0 km/h")
adas_status_label = tk.Label(root, text="ADAS System: Inactive")
adas_status_label.pack()
speed_label.pack()


def update_distances():
    global distances
    while not stop_threads.is_set():
        keys_to_update = []

        if modes['Drive']:
            keys_to_update = ['front', 'frontleft', 'frontright', 'backleft', 'backright']
        elif modes['Reverse']:
            keys_to_update = ['back', 'backleft', 'backright']
        elif modes['Park']:
            keys_to_update = ['backleft', 'backright']

        for key in keys_to_update:
            distances[key] = random.randint(50, 100)

        time.sleep(0.1)  # Sleep for 0.1 seconds

def start_distance_thread():
    global threads_started
    if not threads_started:
        threads_started = True
        stop_threads.clear()
        thread = threading.Thread(target=update_distances)
        thread.daemon = True
        thread.start()


def handle_key_events():
    global modes, direction
    key_pressed = None
    while True:
        if keyboard.is_pressed('f') and key_pressed != 'f':
            modes.update({'Drive': True, 'Reverse': False, 'Park': False, 'Cruise': False})
            start_distance_thread()
            key_pressed = 'f'
        elif keyboard.is_pressed('p') and key_pressed != 'p':
            modes.update({'Drive': False, 'Reverse': False, 'Park': True, 'Cruise': False})
            start_distance_thread()
            key_pressed = 'p'
        elif keyboard.is_pressed('r') and key_pressed != 'r':
            modes.update({'Drive': False, 'Reverse': True, 'Park': False, 'Cruise': False})
            start_distance_thread()
            key_pressed = 'r'
        elif keyboard.is_pressed('c') and key_pressed != 'c':
            modes.update({'Drive': False, 'Reverse': False, 'Park': False, 'Cruise': True})
            start_distance_thread()
            key_pressed = 'c'
        elif not any(keyboard.is_pressed(k) for k in ['f', 'p', 'r', 'c']):
            key_pressed = None

        # Check if gear is in Drive or Reverse and capture WASD keys
        if modes['Drive']:
            if keyboard.is_pressed('w'):
                direction = 'Forward'
            elif keyboard.is_pressed('s'):
                direction = 'Backward'
            elif keyboard.is_pressed('a'):
                direction = 'Left'
            elif keyboard.is_pressed('d'):
                direction = 'Right'
            else:
                direction = None
        elif modes['Reverse']:
            if keyboard.is_pressed('s'):
                direction = 'Backward'
            elif keyboard.is_pressed('a'):
                direction = 'Left'
            elif keyboard.is_pressed('d'):
                direction = 'Right'
            else:
                direction = None
        else:
            direction = None

        time.sleep(0.1)  # Short delay to avoid high CPU usage

def btcontroller():
    # Simulate a Bluetooth controller function that returns connection status and data
    return 1

def print_sensor_data():
    if debug_mode:
        canvas.delete("all")  # Clear previous sensor data
        y_offset = 20
        keys_to_display = []

        if modes['Drive']:
            keys_to_display = ['front', 'frontleft', 'frontright', 'backleft', 'backright']
        elif modes['Reverse']:
            keys_to_display = ['back', 'backleft', 'backright']
        elif modes['Park']:
            keys_to_display = ['backleft', 'backright']

        for key in keys_to_display:
            canvas.create_text(200, y_offset, text=f"Distance from {key}: {distances[key]:.2f} cm", font=("Arial", 12))
            y_offset += 30

        root.update_idletasks()

def update_gear_display():
    current_gear = next((gear for gear, active in modes.items() if active), 'N/A')
    gear_label.config(text=f"Current Gear: {current_gear}")
    root.update_idletasks()

def update_battery_display():
    if modes['Park']:
        battery_level = Park.Park(distances['backleft'], distances['backright'])  # Get battery percentage from Park module
        battery_label.config(text=f"Battery: {battery_level:.1f}%")
    else:
        battery_label.config(text="Battery: N/A")
    root.update_idletasks()

def update_speed():
    global speed
    if modes['Park']:
        speed = 0  # Force speed to 0 in Park mode
    elif any(modes.values()):  # Update speed only if a gear is active
        speed = random.randint(5, 120)  # Generate a random dummy speed value
    else:
        speed = 0
    speed_label.config(text=f"Speed: {speed} km/h")

    # Update ADAS status based on speed
    if speed > 10:
        adas_status_label.config(text="ADAS System: Active")
    else:
        adas_status_label.config(text="ADAS System: Inactive")

def main():
    global modes
    print_sensor_data()  # Display sensor data constantly
    update_speed()  # Update speed display

    if modes['Drive']:
        controller_value = btcontroller()  # Get the controller value
        Drive.Drive(distances['front'], distances['frontleft'], distances['frontright'], distances['backleft'], distances['backright'], controller_value, direction)
    elif modes['Reverse']:
        controller_value = btcontroller()
        Reverse.Reverse(distances['backleft'], distances['backright'], distances['back'], controller_value, direction)
    elif modes['Park']:
        battery_level = Park.Park(distances['backleft'], distances['backright'])  # Get battery percentage from Park module
        battery_label.config(text=f"Battery: {battery_level:.1f}%")
        root.update_idletasks()
    elif modes['Reverse']:
        controller_value = btcontroller()
        Reverse.Reverse(distances['backleft'], distances['backright'], distances['back'], controller_value, direction)
    elif modes['Park']:
        Park.Park(distances['backleft'], distances['backright'])
    elif modes['Cruise']:
        Cruise.Cruise()

def periodic_update():
    main()
    update_gear_display()
    update_battery_display()
    root.after(100, periodic_update)  # Schedule the next update after 0.1 seconds

if __name__ == "__main__":
    # Start the distance update thread immediately for the initial gear
    start_distance_thread()

    # Start the key event handling in a separate thread
    key_event_thread = threading.Thread(target=handle_key_events, daemon=True)
    key_event_thread.start()

    # Start the periodic update
    periodic_update()

    # Start Tkinter main loop
    root.mainloop()

