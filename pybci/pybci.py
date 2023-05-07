from Trainer.TrainBCI import TrainBCI
from LSLScanner import LSLScanner

class PyBCI:
    printDebug = True   # boolean, used to toggle print statements from LSLScanner class

    def __init__(self, dataStreams = None, markerStream= None, streamTypes = None, markerTypes = None, printDebug = True):
        self.lslScanner = LSLScanner(self, dataStreams, markerStream,streamTypes, markerTypes)
        self.printDebug = printDebug
        if self.lslScanner.CheckAvailableLSL():
            self.ConfigureEpochWindowSettings() 

    def __enter__(self):
        # open the resource and return it
        self.StartTraining()
        #self.resource = open("my_file.txt", "r")
        #return self.resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close the resource
        self.StopTraining()

    def StartTraining(self):
        if self.lslScanner.CheckAvailableLSL():
            self.trainer.StartTraining()

    def ConfigureEpochWindowSettings(self, globalWindowSettings = None, customWindowSettings = {}):
        """allows globalWindowSettings to be modified, customWindowSettings is a dict with value names for marker strings which will appear on avalable markerStreams """
        if self.lslScanner.CheckAvailableLSL(): 
            self.trainer = TrainBCI(self, self.lslScanner.dataStreams, self.lslScanner.markerStreams[0], globalWindowSettings, customWindowSettings)

    def ConfigureDataStreamChannels(self, streamChsDropDict = {}):
        self.trainer.ConfigureDataStreamChannels(streamChsDropDict)

    def StopTraining(self):
        pass

    def StartTesting(self):
        if self.lslScanner.CheckAvailableLSL():
            pass

    def StopTesting(self):
        pass
