import serial

ser = serial.Serial('/dev/ttyACM0', 9600)
while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line.startswith("1"):
            print("on")
        if line.startswith("0"):
            print("off")
