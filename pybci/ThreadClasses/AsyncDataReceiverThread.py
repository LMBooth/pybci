import threading
from collections import deque
import itertools
from bisect import bisect_left
def slice_fifo_by_time(fifo, start_time, end_time):
        """Find the slice of fifo between start_time and end_time using binary search."""
        # separate times and data for easier indexing
        times, data = zip(*fifo)
        # find the index of the first time that is not less than start_time
        start_index = bisect_left(times, start_time)
        # find the index of the first time that is not less than end_time
        end_index = bisect_left(times, end_time)
        # return the slice of data between start_index and end_index
        return data[start_index:end_index]

class AsyncDataReceiverThread(threading.Thread):
    """Responsible for receiving data from accepted LSL outlet, slices samples based on tmin+tmax basis, 
    starts counter for received samples after marker is received in ReceiveMarker. Relies on timestamps to slice array, 
    suspected more computationally intensive then synchronous method.
    """
    startCounting = False
    currentMarker = ""
    def __init__(self, closeEvent, trainTestEvent, dataQueueTrain,dataQueueTest, dataStreamInlet,  
                 customEpochSettings, globalEpochSettings,devCount,  streamChsDropDict = [], maxExpectedSampleRate=100):
        # large maxExpectedSampleRate can incur marker drop and slow procesing times for slicing arrays
        super().__init__()
        self.trainTestEvent = trainTestEvent
        self.closeEvent = closeEvent
        self.dataQueueTrain = dataQueueTrain
        self.dataQueueTest = dataQueueTest
        self.dataStreamInlet = dataStreamInlet
        self.customEpochSettings = customEpochSettings
        self.globalEpochSettings = globalEpochSettings
        self.streamChsDropDict = streamChsDropDict
        self.sr = maxExpectedSampleRate
        #self.dataType = dataStreamInlet.info().type()
        self.devCount = devCount # used for tracking which device is sending data to feature extractor
        
    def run(self):
        posCount = 0
        chCount = self.dataStreamInlet.info().channel_count()
        maxTime = (self.globalEpochSettings.tmin + self.globalEpochSettings.tmax)
        if len(self.customEpochSettings.keys())>0:
            if  max([self.customEpochSettings[x].tmin + self.customEpochSettings[x].tmax for x in self.customEpochSettings]) > maxTime:
                maxTime = max([self.customEpochSettings[x].tmin + self.customEpochSettings[x].tmax for x in self.customEpochSettings])
        fifoLength = int(self.sr*maxTime)
        window_end_time = 0
        dataFIFOs = [deque(maxlen=fifoLength) for ch in range(chCount - len(self.streamChsDropDict))]
        while not self.closeEvent.is_set():
            sample, timestamp = self.dataStreamInlet.pull_sample(timeout = 1)
            if sample != None:
                for index in sorted(self.streamChsDropDict, reverse=True):
                    del sample[index] # remove the desired channels from the sample
                for i,fifo in enumerate(dataFIFOs):
                    fifo.append((timestamp, sample[i]))
                if self.trainTestEvent.is_set(): # We're training!
                    if self.startCounting: # we received a marker
                        posCount += 1
                        if posCount >= self.desiredCount:  # enough samples are in FIFO, chop up and put in dataqueue
                            if len(self.customEpochSettings.keys())>0: #  custom marker received
                                if self.customEpochSettings[self.currentMarker].splitCheck: # slice epochs into overlapping time windows
                                    window_length = self.customEpochSettings[self.currentMarker].windowLength
                                    window_overlap = self.customEpochSettings[self.currentMarker].windowOverlap
                                    window_start_time = self.markerTimestamp + self.customEpochSettings[self.currentMarker].tmin
                                    window_end_time = window_start_time + window_length
                                    while window_end_time <= self.markerTimestamp + self.customEpochSettings[self.currentMarker].tmax:
                                        sliceDataFIFOs = [slice_fifo_by_time(fifo, window_start_time, window_end_time) for fifo in dataFIFOs]
                                        self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.devCount])
                                        window_start_time += window_length * (1 - window_overlap)
                                        window_end_time = window_start_time + window_length
                                else: # don't slice just take tmin to tmax time
                                    start_time = self.markerTimestamp + self.customEpochSettings[self.currentMarker].tmin
                                    end_time = self.markerTimestamp + self.customEpochSettings[self.currentMarker].tmax
                                    sliceDataFIFOs = [slice_fifo_by_time(fifo, start_time, end_time) for fifo in dataFIFOs]
                                    self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.devCount])
                            else:
                                if self.globalEpochSettings.splitCheck: # slice epochs in to overlapping time windows
                                    window_length = self.globalEpochSettings.windowLength
                                    window_overlap = self.globalEpochSettings.windowOverlap
                                    window_start_time = self.markerTimestamp - self.globalEpochSettings.tmin
                                    window_end_time = window_start_time + window_length
                                    while window_end_time <= self.markerTimestamp + self.globalEpochSettings.tmax:
                                        sliceDataFIFOs = [slice_fifo_by_time(fifo, window_start_time, window_end_time) for fifo in dataFIFOs]
                                        self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.devCount])
                                        window_start_time += window_length * (1 - window_overlap)
                                        window_end_time = window_start_time + window_length
                                    self.startCounting = False
                                else: # don't slice just take tmin to tmax time
                                    start_time = self.markerTimestamp + self.globalEpochSettings.tmin
                                    end_time = self.markerTimestamp + self.globalEpochSettings.tmax
                                    sliceDataFIFOs = [slice_fifo_by_time(fifo, start_time, end_time) for fifo in dataFIFOs]
                                    self.dataQueueTrain.put([sliceDataFIFOs, self.currentMarker, self.sr, self.devCount])
                            # reset flags and counters
                            posCount = 0
                            self.startCounting = False
                else: # in Test mode
                    if self.globalEpochSettings.splitCheck:
                        window_length = self.globalEpochSettings.windowLength
                        window_overlap = self.globalEpochSettings.windowOverlap
                    else:
                        window_length = self.globalEpochSettings.tmin+self.globalEpochSettings.tmax
            
                    if timestamp >= window_end_time:
                        #sliceDataFIFOs = [[data for time, data in fifo if window_end_time - window_length <= time < window_end_time] for fifo in dataFIFOs]
                        sliceDataFIFOs = [slice_fifo_by_time(fifo, window_end_time - window_length, window_end_time) for fifo in dataFIFOs]
                        self.dataQueueTest.put([sliceDataFIFOs, None, self.devCount])
                        #sliceDataFIFOs = [list(itertools.islice(d, fifoLength-window_samples, fifoLength)) for d in dataFIFOs]
                        if self.globalEpochSettings.splitCheck:
                            window_end_time += window_length * (1 - window_overlap)
                        else:
                            window_end_time = timestamp + window_length
            else:
                pass
                # add levels of debug 
                # print("PyBCI: LSL pull_sample timed out, no data on stream...")


    def ReceiveMarker(self, marker, timestamp): # timestamp will be used for non sample rate specific devices (pupil-labs gazedata)
        #print(marker)
        if self.startCounting == False: # only one marker at a time allow, other in windowed timeframe ignored
            self.currentMarker = marker
            self.markerTimestamp = timestamp
            if len(self.customEpochSettings.keys())>0: #  custom marker received
                if marker in self.customEpochSettings.keys():
                    self.desiredCount = int(self.customEpochSettings[marker].tmax * self.sr) # find number of samples after tmax to finish counting
                    self.startCounting = True
            else: # no custom markers set, use global settings
                self.desiredCount = int(self.globalEpochSettings.tmax * self.sr) # find number of samples after tmax to finish counting
                self.startCounting = True
            #print(self.desiredCount)
