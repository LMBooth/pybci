################ main.py ########################
# Main file for practice lsl markers            #
# Written by Liam Booth 18/02/2021              #
#################################################
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer
import pylsl
import random
from collections import deque
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import numpy as np

class Window(QtWidgets.QWidget):
    commandStrings = ["Fist", "Peace", "Rock", "Shoot", "Thumbs Up"]
    currentMarker = "Fist"
    totchs = 8
    x = deque([x/250 for x in range(250*5)]) #np.zeros(250*5) # this can be edited to change width of x axis 
    y = [deque([0 for x in range(250*5)])  for y in range(totchs)]
    markerOccurred = False
    signal = [deque([0 for x in range(250*5)])  for y in range(totchs)] 
    signalcount = 0
    chunkCount = 0

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        #self.showMaximized()
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
        
        markerInfo = pylsl.StreamInfo("TestMarkers", 'Markers', 1, 0, 'string', 'Limadev')#MB Changed sample rate 
        self.markerOutlet = pylsl.StreamOutlet(markerInfo)
        info = pylsl.StreamInfo("sendTest", 'EMG', self.totchs, 500, 'float32', 'Limadev')
        chns = info.desc().append_child("channels")
        for label in range(self.totchs):
            ch = chns.append_child("channel")
            ch.append_child_value("label", str(label+1))
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
        self.outlet = pylsl.StreamOutlet(info)
        self.ChangeString()
        
    def update(self):
        [self.y[0].popleft() for d in range(50)] 
        markSignal = 1
        for i in range(50):
            if self.markerOccurred:
                markSignal = self.GetMarkerSignal()
            noise = np.random.normal(0, 1, 1)[0]/30
            num = (np.sin(i*0.01*markSignal))# * np.sin(i*0.02))
            #num = (np.sin((self.signalcount+i)*0.03*markSignal) * np.cos((self.signalcount+i)*0.1*markSignal))/2 + noise #+ markSignal
            self.outlet.push_sample([num,num/2,num*2,num*3,num,num,num,num])
            self.y[0].extend([num])#self.signal[0]:self.signalcount+50])
            if self.markerOccurred:
                self.line.setValue((1250-self.chunkCount*50)/250)
        if self.markerOccurred:
            self.chunkCount += 1
            if self.chunkCount > 3:
                self.chunkCount = 0;
                self.markerOccurred = False
        self.signalcount += 50
        if self.signalcount > 1250:
            self.signalcount = 0
        self.plot.setData(list(self.x),list(self.y[0]), pen=pg.mkPen(1), clear=True)#pen = self.colours[x], 

    def GetMarkerSignal(self):   #["Fist", "Peace", "Rock", "Shoot", "Thumbs Up"]
        if self.currentMarker == self.commandStrings[0]:
            markSignal = 2#random.uniform(3, 4)
        elif self.currentMarker == self.commandStrings[1]:
            markSignal = 4#random.uniform(1, 2)
        elif self.currentMarker == self.commandStrings[2]:
            markSignal = 6#random.uniform(0.2, 0.3)
        elif self.currentMarker == self.commandStrings[3]:
            markSignal = 8#random.uniform(0.5, 0.7)
        elif self.currentMarker == self.commandStrings[4]:
            markSignal =  10#random.uniform(6, 7)
        self.savedSignal =  markSignal   
        return markSignal
        
    def BeginTest(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.setInterval(100) 
        self.timer.start()
        self.button.disconnect()
        self.button.setText("Stop Test")
        self.button.clicked.connect(self.StopTest)
        
    def StopTest(self):    
        self.timer.stop()
        self.button.disconnect()
        self.button.setText("Start Test")
        self.button.clicked.connect(self.BeginTest)
        
    def SendMarker(self):   
        self.markerOutlet.push_sample([self.currentMarker])
        self.markerOccurred = True
    
    def SendBaseline(self):
        self.markerOutlet.push_sample(["Baseline"])
        
    def ChangeString(self):
        if not self.markerOccurred:
            self.currentMarker = random.choices(population=self.commandStrings, k=1)[0]
            self.text.setText(self.currentMarker)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())