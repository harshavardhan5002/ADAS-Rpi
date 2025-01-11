import random

def readDistance(sensor):
    # Simulate reading a distance from a sensor
    return random.uniform(0, 100)

def analogRead(pin):
    # Simulate reading an analog value from a sensor
    return random.randint(0, 1023)

def analogWrite(pin, value):
    # Simulate writing a value to an analog pin
    print(f"Analog Write: Pin {pin}, Value {value}")

def digitalWrite(pin, value):
    # Simulate writing a digital value to a pin
    print(f"Digital Write: Pin {pin}, Value {value}")

def map_value(value, fromLow, fromHigh, toLow, toHigh):
    # Map a value from one range to another
    return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow

def clamp(value, min_value, max_value):
    # Clamp a value to be within min_value and max_value
    return max(min_value, min(value, max_value))

def Park(backleft, backright):
    # Example constants for distances and voltage
    dist1 = 30
    dist2 = 20
    dist3 = 10
    emptyVoltage = 3.0
    fullVoltage = 4.2
    voltageSensorPin = 0
    voltageSensitivity = 5.0 / 1024  # Example for 5V and 10-bit ADC

    # Print the distances
    print(f"Distance from Sensor 1011 backleft pin: {backleft:.2f} cm")
    print(f"Distance from Sensor 1213 backright pin: {backright:.2f} cm")

    # BACK LEFT
    if backleft > dist1:
        print("Back Left Ext")
        print("Sending command 6 to slave")
    elif dist1 >= backleft > dist2:
        print("Back Left1")
        print("Sending command 7 to slave")
    elif backleft <= dist2:
        print("Back Left2")
        print("Sending command 8 to slave")

    # BACK RIGHT
    if backright > dist1:
        print("Back Right Ext")
    elif dist1 >= backright > dist2:
        print("Back Right1")
        print("Sending command 7 to slave")
    elif backright <= dist2:
        print("Back Right2")
        print("Sending command 8 to slave")

    # Battery Voltage
    rawValue = analogRead(voltageSensorPin)
    voltage = rawValue * voltageSensitivity

    # Calculate battery percentage
    batteryPercentage = map_value(voltage * 1000, emptyVoltage * 1000, fullVoltage * 1000, 0, 100)

    # Clamp battery percentage to be within 0 to 100
    batteryPercentage = clamp(batteryPercentage, 0, 100)

    # Print battery status
    print(f"Gear : P")
    if batteryPercentage > 0:
        numBars = int(map_value(batteryPercentage, 0, 100, 0, 10))  # Correct integer number of bars
        battery_bar = "[" + "=" * numBars + " " * (10 - numBars) + "]"
        print(f"{battery_bar} {batteryPercentage:.1f}%")
    else:
        print("Battery error")

    # Simulate motor control
    analogWrite('motor1E', 225)
    analogWrite('motor2E', 225)
    analogWrite('motor1I1', 0)
    digitalWrite('motor1I2', 0)
    analogWrite('motor2I1', 0)
    digitalWrite('motor2I2', 0)

    return batteryPercentage  # Return the battery percentage
