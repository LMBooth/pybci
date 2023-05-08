from Utils.LSLScanner import LSLScanner
from ThreadClasses.ThreadControlFunctions import DataReceiverThread, MarkerReceiverThread, FeatureProcessorThread
import queue
import threading

class PyBCI:
    """The PyBCI object stores data from available lsl time series data streams (EEG, pupilometry, EMG, etc.)
     and holds a configerable number of samples based on lsl marker strings.
     If no marker strings are available on the LSL the class will close and return an error.
     Optional Inputs:
        dataStreams = List of strings, allows user to set custom acceptable EEG stream definitions, if None defaults to streamTypes scan
        markerStream = List of strings, allows user to set custom acceptable Marker stream definitions, if None defaults to markerTypes scan
        streamTypes = List of strings, allows user to set custom acceptable EEG type definitions, ignored if dataStreams not None
        markerTypes = List of strings, allows user to set custom acceptable Marker type definitions, ignored if markerStream not None
        printDebug = boolean, if true prints LSLScanner debug information
    """
    printDebug = True   # boolean, used to toggle print statements from LSLScanner class
    globalWindowSettings = [
        True,   #addcheck,  checks whether or not to include the epoch
        False,  #splitcheck, checks whether or not subdivide epochs
        0,      #tmins, time in seconds to capture samples before trigger
        1,      #tmaxs, time in seconds to capture samples after trigger
        0.5,    #window_lengths, if splitcheck true - time in seconds to split epoch
        0.5     #overlaps, percentage value > 0 and < 1, example if epoch has tmin of 0 and tmax of 1 with window 
        # length of 0.5 we have 1 epoch between t 0 and t0.5 another at 0.25 to 0.75, 0.5 to 1
     ] 
    markerThread = []
    dataThreads = []
    streamChsDropDict= {}
    dataStreams = []
    markerStream = None

    def __init__(self, dataStreams = None, markerStream= None, streamTypes = None, markerTypes = None, printDebug = True):
        self.lslScanner = LSLScanner(self, dataStreams, markerStream,streamTypes, markerTypes)
        self.printDebug = printDebug
        if self.lslScanner.CheckAvailableLSL() == True:
            self.ConfigureEpochWindowSettings() 
            self.__StartThreads()

    def __enter__(self): # with bci
        if self.lslScanner.CheckAvailableLSL():
            self.ConfigureEpochWindowSettings()
            self.__StartThreads()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.StopThreads()

    def StartTraining(self):
        if self.lslScanner.CheckAvailableLSL():
            pass
            #self.trainer.StartTraining()

    def StopTraining(self):
        pass
        #self.trainer.StopTraining()

    def StartTesting(self):
        if self.lslScanner.CheckAvailableLSL():
            pass

    def StopTesting(self):
        pass

    def __StartThreads(self):
        self.featureQueue = queue.Queue()
        self.dataQueue = queue.Queue()
        totalDevices = len(self.dataStreams)
        lock = threading.Lock() # used for printing in threads
        self.closeEvent = threading.Event() # used for closing threads
        if self.printDebug:
            print("PyLSL-BCI: Starting threads initialisation...")
        self.featureThread = FeatureProcessorThread(self.closeEvent, self.dataQueue, self.featureQueue, totalDevices, lock, self.customWindowSettings)
        self.featureThread.start()
        self.dataThreads = []
        for stream in self.dataStreams:
            if stream.info().name() in self.streamChsDropDict.keys():
                dt = DataReceiverThread(self.closeEvent, self.dataQueue, stream,  self.customWindowSettings, self.globalWindowSettings, self.streamChsDropDict[stream.info().name()])
            else:
                dt = DataReceiverThread(self.closeEvent, self.dataQueue, stream,  self.customWindowSettings, self.globalWindowSettings)
            dt.start()
            self.dataThreads.append(dt)
        self.markerThread = MarkerReceiverThread(self.closeEvent, self.markerStream,self.dataThreads, self.featureThread)
        self.markerThread.start()


    def StopThreads(self):
        self.closeEvent.set()
        # wait for all threads to finish processing, probably worth pulling out finalised classifier information stored for later use.
        for dt in self.dataThreads:
            dt.thread.join()
        self.markerThread.join()
        self.featureThread.join()



    # Could move all configures to a configuration class, might make options into more descriptive classes?
    def ConfigureEpochWindowSettings(self, globalWindowSettings = None, customWindowSettings = {}):
        """allows globalWindowSettings to be modified, customWindowSettings is a dict with value names for marker strings which will appear on avalable markerStreams """
        self.customWindowSettings = customWindowSettings
        if globalWindowSettings != None:
            self.globalWindowSettings = globalWindowSettings

    def ConfigureFeatures(self,freqbands = None, featureChoices = None ):
        # potentially should move configuration to generic class which can be used for both test and train
        if freqbands != None:
            self.featureThread.freqbands = freqbands
        if featureChoices != None:    
            self.featureThread.featureChoices = featureChoices

    def ConfigureDataStreamChannels(self,streamChsDropDict = {}):
        # potentially should move configuration to generic class which can be used for both test and train
        self.streamChsDropDict = streamChsDropDict 