################ main.py ########################
# Main file for practice lsl markers            #
# Please note! sample rate is not exact,        #
# expect some drop over time!                   #
# Written by Liam Booth 14/05/2023              #
#################################################
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer
import pylsl
import random, time
from collections import deque
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import numpy as np
import threading

def precise_sleep(duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        pass

class PsuedoEMGDataConfig:
    duration = 1.0 
    noise_level = 0.1
    amplitude = 0.2
    frequency = 1.0

class Window(QtWidgets.QWidget):
    commandStrings = ["Marker1", "Marker2", "Marker3", "Marker4", "Marker5"]
    commandDataConfigs = [PsuedoEMGDataConfig(), PsuedoEMGDataConfig(), PsuedoEMGDataConfig(), PsuedoEMGDataConfig(),PsuedoEMGDataConfig()]
    baselineConfig  = PsuedoEMGDataConfig()
    currentMarker = "Marker1"
    totchs = 8
    x = deque([x/250 for x in range(250*5)]) #np.zeros(250*5) # this can be edited to change width of x axis 
    y = [deque([0 for x in range(250*5)])  for y in range(totchs)]
    markerOccurred = False
    signal = [deque([0 for x in range(250*5)])  for y in range(totchs)] 
    signalcount = 0
    chunkCount = 0
    sampleRate = 250
    #de = deque(maxlen=250/10) maybe make a timestap deque to keep consistent timing?

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        #self.showMaximized()
        self.lock = threading.Lock()  # Lock for thread safety
        # setting up different signal varables
        self.commandDataConfigs[0].amplitude = 1
        self.commandDataConfigs[0].noise_level = 1
        self.commandDataConfigs[0].frequency = 2
        self.commandDataConfigs[1].amplitude = 1
        self.commandDataConfigs[1].noise_level = 0.5
        self.commandDataConfigs[1].frequency = 5
        self.commandDataConfigs[2].amplitude = 2
        self.commandDataConfigs[2].noise_level = 0.3
        self.commandDataConfigs[2].frequency = 15
        self.commandDataConfigs[3].amplitude = 2
        self.commandDataConfigs[3].noise_level = 0.6
        self.commandDataConfigs[3].frequency = 0.2
        self.commandDataConfigs[4].amplitude = 3
        self.commandDataConfigs[4].noise_level = 0.1
        self.commandDataConfigs[4].frequency = 2

        self.button = QtWidgets.QPushButton('Start Signal Test')
        self.button.clicked.connect(self.BeginTest)
        self.button2 = QtWidgets.QPushButton('Change Marker')
        self.button2.clicked.connect(self.ChangeString)
        self.button3 = QtWidgets.QPushButton('Send Marker')
        self.button3.clicked.connect(self.SendMarker)
        self.buttonBaseline = QtWidgets.QPushButton('Send Baseline')
        self.buttonBaseline.clicked.connect(self.SendBaseline)
        self.text = QtWidgets.QLabel("Command!")
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        self.text.setFont(QFont('Times', 50))
        layout = QtWidgets.QVBoxLayout(self)
        self.graphWidget = pg.PlotWidget()
        self.line = pg.InfiniteLine(pen = pg.mkPen('b',width=3))
        self.graphWidget.addItem(self.line)
        layout.addWidget(self.graphWidget)
        layout.addWidget(self.button)
        layout.addWidget(self.text)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.buttonBaseline)
        self.plot = self.graphWidget.plot()
        
        markerInfo = pylsl.StreamInfo("TestMarkers", 'Markers', 1, 0, 'string', 'Dev')#MB Changed sample rate 
        self.markerOutlet = pylsl.StreamOutlet(markerInfo)
        info = pylsl.StreamInfo("sendTest", 'EMG', self.totchs, self.sampleRate, 'float32', 'Dev')
        chns = info.desc().append_child("channels")
        for label in range(self.totchs):
            ch = chns.append_child("channel")
            ch.append_child_value("label", str(label+1))
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
        self.outlet = pylsl.StreamOutlet(info)
        self.ChangeString()
        
    def update(self):
        with self.lock:  # Acquire the lock
            if self.markerOccurred:
                for i, command in enumerate(self.commandStrings):
                    if self.currentMarker == command:
                        num_samples = int(self.commandDataConfigs[i].duration/10 * self.sampleRate)
                        [self.y[0].popleft() for d in range(num_samples)] 
                        num = self.GeneratePseudoEMG(self.sampleRate,self.commandDataConfigs[i].duration/10, self.commandDataConfigs[i].noise_level, 
                                                        self.commandDataConfigs[i].amplitude, self.commandDataConfigs[i].frequency)
                self.chunkCount += 1
                if self.chunkCount >= 10:
                    self.markerOccurred = False
                    self.chunkCount = 0 
            else:# send baseline
                num_samples = int(self.baselineConfig.duration/10 * self.sampleRate)
                [self.y[0].popleft() for d in range(num_samples)] 
                num = self.GeneratePseudoEMG(self.sampleRate,self.baselineConfig.duration/10, self.baselineConfig.noise_level, 
                                                self.baselineConfig.amplitude, self.baselineConfig.frequency)
            [self.y[0].extend([num[0][d]]) for d in range(num_samples)] 
            for n in range(num.shape[1]):
                self.outlet.push_sample(num[:,n])
            #print(num.T.shape)
            #chunk = num.T
            #print(chunk)
            #for sample in chunk:
            #    if len(sample) != 8:
            #        print(f"Invalid sample: {sample}")
            #self.outlet.push_chunk(chunk)
            self.plot.setData(list(self.x),list(self.y[0]), pen=pg.mkPen(1), clear=True)#pen = self.colours[x], 

    def GeneratePseudoEMG(self,samplingRate, duration, noise_level, amplitude, frequency):
        """
        Generate a pseudo EMG signal for a given gesture.
        Arguments:
        - sampling_rate: Number of samples per second
        - duration: Duration of the signal in seconds
        - noise_level: The amplitude of Gaussian noise to be added (default: 0.1)
        - amplitude: The amplitude of the EMG signal (default: 1.0)
        - frequency: The frequency of the EMG signal in Hz (default: 10.0)
        Returns:
        - emg_signal: The generated pseudo EMG signal as a 2D numpy array with shape (channels, samples)
        """
        num_samples = int(duration * samplingRate)
        # Initialize the EMG signal array
        emg_signal = np.zeros((self.totchs, num_samples))
        # Generate the pseudo EMG signal for each channel
        for channel in range(self.totchs):
            # Calculate the time values for the channel
            times = np.linspace(0, duration, num_samples)
            # Generate the pseudo EMG signal based on the selected gesture
            emg_channel = amplitude * np.sin(2 * np.pi * frequency * times)# * times)  # Sinusoidal EMG signal
            # Add Gaussian noise to the EMG signal
            noise = np.random.normal(0, noise_level, num_samples)
            emg_channel += noise
            # Store the generated channel in the EMG signal array
            emg_signal[channel, :] = emg_channel
        return emg_signal

    def BeginTest(self):
        self.stop_signal = False
        self.thread = threading.Thread(target=self.generate_signal)
        self.thread.start()
        self.button.disconnect()
        self.button.setText("Stop Test")
        self.button.clicked.connect(self.StopTest)
    
    def generate_signal(self):
        while not self.stop_signal:
            start_time = time.time()
            self.update()
            sleep_duration = max(0, (1.0 / 10) - (start_time - time.time()))
            precise_sleep(sleep_duration)

    def StopTest(self):    
        self.stop_signal = True
        self.thread.join() # wait for the thread to finish
        self.button.disconnect()
        self.button.setText("Start Test")
        self.button.clicked.connect(self.BeginTest)
        
    def SendMarker(self):
        with self.lock:  # Acquire the lock
            self.markerOutlet.push_sample([self.currentMarker])
            self.markerOccurred = True
    
    def SendBaseline(self):
        self.markerOutlet.push_sample(["Baseline"])
        
    def ChangeString(self):
        if not self.markerOccurred:
            newPos = self.commandStrings.index(self.currentMarker)+1
            if newPos >= len(self.commandStrings):
                newPos = 0
            self.currentMarker = self.commandStrings[newPos] # = random.choices(population=self.commandStrings, k=1)[0]
            self.text.setText(self.currentMarker)
    
    def closeEvent(self, event):
        # Stop the worker thread when the widget is closed
        self.stop_signal = True
        self.thread.join() # wait for the thread to finish
        self.button.disconnect()

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())