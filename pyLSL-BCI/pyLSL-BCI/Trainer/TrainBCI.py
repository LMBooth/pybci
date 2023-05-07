import queue
import threading
from ThreadControlFunctions import DataReceiverThread, MarkerReceiverThread

class TrainBCI():
    """
    The TrainBCI object stores data from available lsl time series data streams (EEG, pupilometry, EMG, etc.)
     and holds a configerable number of samples based on lsl marker strings.
     If no marker strings are available on the LSL the class will close and return an error.
     
    A sklearn clf of tensorflow model can be passed or sklearns svm wil be used by default.
    
    EEG, ECG EMG and pupilemetry streams all have togglable custom 'features' which can be used.
    Future work: maybe adapt to allow for of raw time-series data.
    """
    globalWindowSettings = [
        True,   #addcheck,  checks whether or not to include the epoch
        False,  #splitcheck, checks whether or not subdivide epochs
        0,      #tmins, time in seconds to capture samples before trigger
        1,      #tmaxs, time in seconds to capture samples after trigger
        0.5,    #window_lengths, if splitcheck true - time in seconds to split epoch
        0.5     #overlaps, percentage value > 0 and < 1, example if epoch has tmin of 0 and tmax of 1 with window length of 0.5 we have 1 epoch between t 0 and t0.5 another at 0.25 to 0.75, 0.5 to 1
     ] 
    markerThread = []
    dataThreads = []
    streamChsDropDict= {}
    def __init__(self, parent, dataStreams, markerStream, globalWindowSettings = None, customWindowSettings = {}):
        super().__init__()
        self.parent = parent
        self.dataStreams = dataStreams
        self.markerStream = markerStream
        if globalWindowSettings != None:
            self.globalWindowSettings = globalWindowSettings
        self.customWindowSettings = customWindowSettings
        self.samples = {stream_name: [] for stream_name in dataStreams}
        self.ConfigureDataStreamChannels()
        self.featureThread = FeatureProcessorThread()
        self.featureThread.start()
        
    def ConfigureFeatures(self,freqbands = None, featureChoices = None ):
        # potentially should move configuration to generic class which can be used for both test and train
        self.featureThread.freqbands = freqbands
        self.featureThread.featureChoices = featureChoices

    def ConfigureDataStreamChannels(self,streamChsDropDict = {}):
        # potentially should move configuration to generic class which can be used for both test and train
        self.streamChsDropDict = streamChsDropDict 

    def StartTraining(self):
        self.markerQueue = queue.Queue()
        self.dataQueue = queue.Queue()
        lock = threading.Lock()
        if self.parent.printDebug:
            print("PyLSL-BCI: Starting threads")
        self.dataThreads = []
        for stream in self.dataStreams:
            if stream.info().name() in self.streamChsDropDict.keys():
                dt = DataReceiverThread(self.markerQueue, self.dataQueue, stream,  self.customWindowSettings, self.globalWindowSettings, self.streamChsDropDict[stream.info().name()])
            else:
                dt = DataReceiverThread(self.markerQueue, self.dataQueue, stream,  self.customWindowSettings, self.globalWindowSettings)
            dt.start()
            self.dataThreads.append(dt)
        self.markerThread = MarkerReceiverThread(self.markerQueue, self.markerStream,self.dataThreads)
        self.markerThread.start()


class EEGFeatureProcessor():
    pass
