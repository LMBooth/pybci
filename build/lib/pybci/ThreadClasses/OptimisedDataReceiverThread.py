import threading, time
from collections import deque
import itertools
import numpy as np
from bisect import bisect_left

class OptimisedDataReceiverThread(threading.Thread):
    """Responsible for receiving data from accepted LSL outlet, slices samples based on tmin+tmax basis, 
    starts counter for received samples after marker is received in ReceiveMarker. Relies on timestamps to slice array, 
    suspected more computationally intensive then synchronous method.
    """
    markerReceived = False
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
        self.devCount = devCount # used for tracking which device is sending data to feature extractor
        
    def run(self):
        chCount = self.dataStreamInlet.info().channel_count()
        maxTime = (self.globalEpochSettings.tmin + self.globalEpochSettings.tmax)
        if len(self.customEpochSettings.keys()) > 0: # min time used for max_samples and temp array,  maxTime used for longest epochs and permanentDataBuffers
            if max([self.customEpochSettings[x].tmin + self.customEpochSettings[x].tmax for x in self.customEpochSettings]) > maxTime:
                maxTime = max([self.customEpochSettings[x].tmin + self.customEpochSettings[x].tmax for x in self.customEpochSettings])
        if self.globalEpochSettings.splitCheck:
            window_length = self.globalEpochSettings.windowLength
            window_overlap = self.globalEpochSettings.windowOverlap
        else:
            window_length = self.globalEpochSettings.tmin+self.globalEpochSettings.tmax
        minfifoLength = int(self.sr * window_length * (1-self.globalEpochSettings.windowOverlap)) #  sets global window length with overlap as factor for minimum delay in test mode
        dataBuffers = np.zeros((minfifoLength,chCount))
        chs_to_drop = np.ones(chCount, dtype=bool)
        chs_to_drop[self.streamChsDropDict] = False
        fifoLength = int(self.sr * (maxTime+20)) # adds twenty seconds to give more timestamps when buffering  (assuming devices dont timeout for longer then 20.0 seconds, migth be worth making configurable)
        permanentDataBuffers = np.zeros((fifoLength, chCount - len(self.streamChsDropDict)))
        permanentTimestampBuffer = np.zeros(fifoLength)
        next_window_time = 0 # sets testing mode window time, duration based on windowlength and overlap
        while not self.closeEvent.is_set():
            _, timestamps = self.dataStreamInlet.pull_chunk(timeout=0.0, max_samples=dataBuffers.shape[0], dest_obj=dataBuffers) # optimised method of getting data to pull_sample, dest_obj saves memory re-allocation
            if timestamps:
                if len(self.streamChsDropDict) == 0:
                    dataBufferView = dataBuffers[:len(timestamps), :] # [:, :len(timestamps)]
                else:
                    dataBufferView = dataBuffers[:len(timestamps), chs_to_drop] # [:, :len(timestamps)]
                permanentDataBuffers = np.roll(permanentDataBuffers, shift=-len(timestamps), axis=0)
                permanentTimestampBuffer = np.roll(permanentTimestampBuffer, shift=-len(timestamps))
                permanentDataBuffers[-len(timestamps):,:] = dataBufferView
                permanentTimestampBuffer[-len(timestamps):] = timestamps
                if self.trainTestEvent.is_set(): # We're training!
                    if self.markerReceived: # we received a marker 
                        timestamp_tmin = self.targetTimes[1]
                        timestamp_tmax = self.targetTimes[0]
                        #print(timestamp_tmin, "timestamp_tmin >= permanentTimestampBuffer[0]:", permanentTimestampBuffer[0])
                        #print(timestamp_tmax, "timestamp_tmax <=  permanentTimestampBuffer[-1]:", permanentTimestampBuffer[-1])
                        if timestamp_tmin >= permanentTimestampBuffer[0] and timestamp_tmax <= permanentTimestampBuffer[-1]:
                            if len(self.customEpochSettings.keys())>0: #  custom marker received
                                if self.customEpochSettings[self.currentMarker].splitCheck: # slice epochs into overlapping time windows
                                    window_start_time = timestamp_tmin
                                    window_end_time = window_start_time + window_length
                                    while window_end_time <= timestamp_tmax:
                                        idx_tmin = (np.abs(permanentTimestampBuffer - window_start_time)).argmin() # find array index of start of window
                                        idx_tmax = (np.abs(permanentTimestampBuffer - window_end_time)).argmin() # find array index of end of window
                                        slices = permanentDataBuffers[idx_tmin:idx_tmax,:]
                                        self.dataQueueTrain.put([slices, self.currentMarker, self.sr, self.devCount])
                                        window_start_time += window_length * (1 - window_overlap)
                                        window_end_time = window_start_time + window_length
                                else: # don't slice just take tmin to tmax time
                                    idx_tmin = (np.abs(permanentTimestampBuffer - timestamp_tmin)).argmin()
                                    idx_tmax = (np.abs(permanentTimestampBuffer - timestamp_tmax)).argmin()
                                    slices = permanentDataBuffers[idx_tmin:idx_tmax,:]
                                    self.dataQueueTrain.put([slices, self.currentMarker, self.sr, self.devCount])
                            else:
                                if self.globalEpochSettings.splitCheck: # slice epochs in to overlapping time windows
                                    window_start_time = timestamp_tmin#self.markerTimestamp - self.globalEpochSettings.tmin
                                    window_end_time = window_start_time + window_length
                                    #print(window_end_time, "   ", timestamp_tmax)
                                    while window_end_time <= timestamp_tmax: #:self.markerTimestamp + self.globalEpochSettings.tmax:
                                        idx_tmin = (np.abs(permanentTimestampBuffer - window_start_time)).argmin()
                                        idx_tmax = (np.abs(permanentTimestampBuffer - window_end_time)).argmin()
                                        #print(idx_tmin, "   ", idx_tmax)
                                        #print(permanentTimestampBuffer[idx_tmin], "  ", permanentTimestampBuffer[idx_tmax])
                                        #print(permanentDataBuffers.shape)
                                        slices = permanentDataBuffers[idx_tmin:idx_tmax,:]
                                        #print(slices.shape)
                                        self.dataQueueTrain.put([slices, self.currentMarker, self.sr, self.devCount])
                                        window_start_time += window_length * (1 - window_overlap)
                                        window_end_time = window_start_time + window_length
                                    self.startCounting = False
                                else: # don't slice just take tmin to tmax time
                                    idx_tmin = (np.abs(permanentTimestampBuffer - timestamp_tmin)).argmin()
                                    idx_tmax = (np.abs(permanentTimestampBuffer - timestamp_tmax)).argmin()
                                    slices = permanentDataBuffers[idx_tmin:idx_tmax,:]
                                    self.dataQueueTrain.put([slices, self.currentMarker, self.sr, self.devCount])
                            self.markerReceived = False
                else: # in Test mode
                    if next_window_time+(window_length/2) <= permanentTimestampBuffer[-1]:
                        idx_tmin = (np.abs(permanentTimestampBuffer - (next_window_time-(window_length/2)))).argmin()
                        idx_tmax = (np.abs(permanentTimestampBuffer - (next_window_time+(window_length/2)))).argmin()
                        #print(next_window_time+(window_length/2), "next_window_time+(window_length/2) <= permanentTimestampBuffer[-1]:", permanentTimestampBuffer[-1])
                        #print(next_window_time, "next_window_time   permanentTimestampBuffer[0]:", permanentTimestampBuffer[0])
                        #print(idx_tmin, "idx_tmin   idx_tmax", idx_tmax)
                        if idx_tmin == idx_tmax:
                            # oops we lost track, get window positions and start again
                            idx_tmin = (np.abs(permanentTimestampBuffer - (permanentTimestampBuffer[-1] - window_length))).argmin() 
                            idx_tmax = -1
                            next_window_time = permanentTimestampBuffer[-1] - (window_length/2)
                        slices = permanentDataBuffers[idx_tmin:idx_tmax,:]
                        self.dataQueueTest.put([slices,  self.sr, self.devCount])
                        if self.globalEpochSettings.splitCheck:
                            next_window_time += window_length * (1 - window_overlap)
                        else:
                            next_window_time += window_length
            else:
                pass
                # add levels of debug?

    def ReceiveMarker(self, marker, timestamp): # timestamp will be used for non sample rate specific devices (pupil-labs gazedata)
        if self.markerReceived == False: # only one marker at a time allow, other in windowed timeframe ignored
            self.currentMarker = marker
            self.markerTimestamp = timestamp
            if len(self.customEpochSettings.keys())>0: #  custom marker received
                if marker in self.customEpochSettings.keys():
                    #self.desiredCount = int(self.customEpochSettings[marker].tmax * self.sr) # find number of samples after tmax to finish counting
                    self.targetTimes = [timestamp+self.customEpochSettings[marker].tmax, timestamp-self.customEpochSettings[marker].tmin]
                    self.markerReceived = True
            else: # no custom markers set, use global settings
                #self.desiredCount = int(self.globalEpochSettings.tmax * self.sr) # find number of samples after tmax to finish counting
                self.targetTimes = [timestamp+self.globalEpochSettings.tmax, timestamp-self.globalEpochSettings.tmin]
                self.markerReceived = True