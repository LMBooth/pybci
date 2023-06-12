import serial
from pylsl import StreamInfo, StreamOutlet,  StreamInlet, resolve_stream
# Setup serial connection
ser = serial.Serial('COM9', 9600)  # change '/dev/ttyACM0' to your serial port name
# Setup LSL
info = StreamInfo('ArduinoHandData', 'EMG', 1, 100, 'float32', 'myuid34234')
outlet = StreamOutlet(info)

# Look for the marker stream
print("looking for a marker stream...")
streams = resolve_stream('name', 'MarkerHandGrasps')

inlet = StreamInlet(streams[0])

ser.write("1\r".encode()) 
print("Beginning transmission...")
while True:
    try:
        # Read data from the Arduino and send it to LSL
        if ser.in_waiting:
            data = ser.readline().strip()  # read a '\n' terminated line, strip newline characters
            #print(data)
            if data:
                try:
                    data = float(data)  # convert data to float
                    outlet.push_sample([data])  # send data to LSL
                except ValueError:
                    pass  # ignore this reading
        # Read data from the LSL marker stream and send it to the Arduino
        marker, timestamp = inlet.pull_sample(0.0)
        if marker:
            grasp_pattern = str(marker[0])
            ser.write(grasp_pattern.encode())  # send grasp pattern to the Arduino
            
    except KeyboardInterrupt:
        break