PyBCI
=====

.. automodule:: PyBCI
   :members:

class PyBCI
-----------

.. autoclass:: PyBCI
   :members:

The PyBCI object stores data from available lsl time series data streams (EEG, pupilometry, EMG, etc.) and holds a configurable number of samples based on lsl marker strings. If no marker strings are available on the LSL, the class will close and return an error.

.. automethod:: __init__(self, dataStreams = None, markerStream= None, streamTypes = None, markerTypes = None, printDebug = True, globalEpochSettings = GlobalEpochSettings(), customEpochSettings = {}, streamChsDropDict = {}, streamCustomFeatureExtract = {}, minimumEpochsRequired = 10, clf= None, model = None)

.. automethod:: __enter__(self, dataStreams = None, markerStream= None, streamTypes = None, markerTypes = None, printDebug = True, globalEpochSettings = GlobalEpochSettings(), customEpochSettings = {}, streamChsDropDict = {}, freqbands = [[1.0, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 20.0]], featureChoices = GeneralFeatureChoices())

.. automethod:: __exit__(self, exc_type, exc_val, exc_tb)

.. automethod:: Connect(self)

.. automethod:: TrainMode(self)

.. automethod:: TestMode(self)

.. automethod:: CurrentClassifierInfo(self)

.. automethod:: CurrentClassifierMarkerGuess(self)

.. automethod:: ReceivedMarkerCount(self)

.. automethod:: StopThreads(self)

.. automethod:: ConfigureMachineLearning(self, minimumEpochsRequired = 10, clf = None, model = None)

.. automethod:: ConfigureEpochWindowSettings(self, globalEpochSettings = GlobalEpochSettings(), customEpochSettings = {})

.. automethod:: ConfigureDataStreamChannels(self, streamChsDropDict = {})

.. automethod:: ResetThreadsAfterConfigs(self)