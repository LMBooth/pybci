import threading
from collections import deque
import itertools

class DataReceiverThread(threading.Thread):
    """Responsible for receiving data from accepted LSL outlet, slices samples based on tmin+tmax basis, 
    starts counter for received samples after marker is received in ReceiveMarker.
    """
    startCounting = False
    currentMarker = ""
    def __init__(self, closeEvent, trainTestEvent, dataQueueTrain,dataQueueTest, dataStreamInlet,  
                 customEpochSettings, globalEpochSettings,devCount,  streamChsDropDict = []):
        super().__init__()
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.dataQueueTrain = dataQueueTrain
        self.dataQueueTest = dataQueueTest
        self.dataStreamInlet = dataStreamInlet
        self.customEpochSettings = customEpochSettings
        self.globalEpochSettings = globalEpochSettings
        self.streamChsDropDict = streamChsDropDict
        self.sr = dataStreamInlet.info().nominal_srate()
        #self.dataType = dataStreamInlet.info().type()
        self.devCount = devCount # used for tracking which device is sending data to feature extractor
        
    def run(self):
        posCount = 0
        chCount = self.dataStreamInlet.info().channel_count()
        maxTime = (self.globalEpochSettings.tmin + self.globalEpochSettings.tmax)
        if len(self.customEpochSettings.keys())>0:
            if  max([self.customEpochSettings[x].tmin + self.customEpochSettings[x].tmax for x in self.customEpochSettings]) > maxTime:
                maxTime = max([self.customEpochSettings[x].tmin + self.customEpochSettings[x].tmax for x in self.customEpochSettings])
        fifoLength = int(self.dataStreamInlet.info().nominal_srate()*maxTime)
        #print(fifoLength)
        dataFIFOs = [deque(maxlen=fifoLength) for ch in range(chCount - len(self.streamChsDropDict))]
        while not self.closeEvent.is_set():
            sample, timestamp = self.dataStreamInlet.pull_sample(timeout = 1)
            if sample != None:
                for index in sorted(self.streamChsDropDict, reverse=True):
                    del sample[index] # remove the desired channels from the sample
                for i,fifo in enumerate(dataFIFOs):
                    fifo.append(sample[i])
                if self.trainTestEvent.is_set(): # We're training!
                    
                    if self.startCounting: # we received a marker
                        posCount+=1
                        if posCount >= self.desiredCount:  # enough samples are in FIFO, chop up and put in dataqueue
                            if len(self.customEpochSettings.keys())>0: #  custom marker received
                                if self.customEpochSettings[self.currentMarker].splitCheck: # slice epochs in to overlapping time windows
                                    window_samples =int(self.customEpochSettings[self.currentMarker].windowLength * self.sr) #number of samples in each window
                                    increment = int((1-self.customEpochSettings[self.currentMarker].windowOverlap)*window_samples) # if windows overlap each other by how many samples
                                    while posCount - window_samples > 0:
                                        sliceDataFIFOs = [list(itertools.islice(d, posCount - window_samples, posCount)) for d in dataFIFOs]
                                        self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr,  self.devCount])
                                        posCount-=increment
                                else: # don't slice just take tmin to tmax time
                                    sliceDataFIFOs = [list(itertools.islice(d, fifoLength - int((self.customEpochSettings[self.currentMarker].tmax+self.customEpochSettings[self.currentMarker].tmin) * self.sr), fifoLength))for d in dataFIFOs]
                                    self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr,  self.devCount])
                            else:
                                if self.globalEpochSettings.splitCheck: # slice epochs in to overlapping time windows
                                    window_samples =int(self.globalEpochSettings.windowLength * self.sr) #number of samples in each window
                                    increment = int((1-self.globalEpochSettings.windowOverlap)*window_samples) # if windows overlap each other by how many samples
                                    startCount = self.desiredCount + int(self.globalEpochSettings.tmin * self.sr)
                                    while startCount - window_samples > 0:
                                        sliceDataFIFOs = [list(itertools.islice(d, startCount - window_samples, startCount)) for d in dataFIFOs]
                                        self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.devCount])
                                        startCount-=increment
                                else: # don't slice just take tmin to tmax time
                                    sliceDataFIFOs = [list(itertools.islice(d, fifoLength - int((self.globalEpochSettings.tmin+self.globalEpochSettings.tmax) * self.sr), fifoLength)) for d in dataFIFOs]
                                    self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.devCount])
                            # reset flags and counters
                            posCount = 0
                            self.startCounting = False
                else: # in Test mode
                    posCount+=1
                    if self.globalEpochSettings.splitCheck:
                        window_samples = int(self.globalEpochSettings.windowLength * self.sr) #number of samples in each window
                    else:
                        window_samples = int((self.globalEpochSettings.tmin+self.globalEpochSettings.tmax) * self.sr)
                    if posCount >= window_samples:
                        sliceDataFIFOs = [list(itertools.islice(d, fifoLength-window_samples, fifoLength)) for d in dataFIFOs]
                        if self.globalEpochSettings.splitCheck:
                            posCount = int((1-self.globalEpochSettings.windowOverlap)*window_samples) # offset poscoutn based on window overlap 
                        else:
                            posCount = 0
                            
                        self.dataQueueTest.put([sliceDataFIFOs, self.sr, self.devCount])
            else:
                pass
                # add levels of debug 
                # print("PyBCI: LSL pull_sample timed out, no data on stream...")

    def ReceiveMarker(self, marker, timestamp): # timestamp will be used for non sample rate specific devices (pupil-labs gazedata)
        #print(marker)
        if self.startCounting == False: # only one marker at a time allow, other in windowed timeframe ignored
            self.currentMarker = marker
            if len(self.customEpochSettings.keys())>0: #  custom marker received
                if marker in self.customEpochSettings.keys():
                    self.desiredCount = int(self.customEpochSettings[marker].tmax * self.sr) # find number of samples after tmax to finish counting
                    self.startCounting = True
            else: # no custom markers set, use global settings
                self.desiredCount = int(self.globalEpochSettings.tmax * self.sr) # find number of samples after tmax to finish counting
                self.startCounting = True
            #print(self.desiredCount)
