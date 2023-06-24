# Arduino hand-grasp example

| File             | Description |
|------------------|-------------|
| ArduinoToLSL.py | Grabs data from arduino via serial port and puts on to LSL data stream, also receives LSL marker stream from testArduinoHand.py to send hand servo motor position commands to the arduino. |
| MarkerMaker.py  | Used to deliver training stimulus via PyQt5 GUI, sends LSL markers to train PyBCI. Receives Markers via LSL from testArduinoHand.py when in test mode. |
| ServoControl.ino | Arduino script used to capture data from myoware sensor and send via serial port. Also, controls servo motors and finger positions from serial commands. |
| testArduinoHand.py | PyBCI script which receives lsl datastream from ArduinoToLSL.py and uses MarkerMaker.py to deliver training stimulus markers. When in test mdoe outputs LSL marker stream to send commands to the arduino via ArduinoToLSL.py. |
| testArduinoPytorch.py | Similar to testArduinoHand.py but using PyTorch. |

This folder contains an example of how to push ADC recordings from an arduino to the LSL, use the PyBCI package to classify the incoming data, then pushes command data back to the arduino in test moded to control servo positions.

This is an extremely simple example setup, using a single [Myoware Sensor](https://myoware.com/products/muscle-sensor/) as an input to an arduino ADC, then the 3D printed finger positions are controlled with pulleys on servomotors. 

An example video can be found here:

[![PyBCI Arduino Hand Demo](https://img.youtube.com/vi/InEbiykeinQ/0.jpg)](https://www.youtube.com/watch?v=InEbiykeinQ)
