PyBCI
=====
.. class:: PyBCI(dataStreams=None, markerStream=None, streamTypes=None, markerTypes=None, printDebug=True, globalEpochSettings=GlobalEpochSettings(), customEpochSettings={}, streamChsDropDict={}, streamCustomFeatureExtract={}, minimumEpochsRequired=10, clf=None, model=None)

The PyBCI object stores data from available LSL time series data streams (EEG, pupilometry, EMG, etc.) and holds a configurable number of samples based on LSL marker strings.

:param list dataStreams: Allows user to set custom acceptable EEG stream definitions, if None defaults to streamTypes scan.
:param list markerStream: Allows user to set custom acceptable Marker stream definitions, if None defaults to markerTypes scan.
:param list streamTypes: Allows user to set custom acceptable EEG type definitions, ignored if dataStreams not None.
:param list markerTypes: Allows user to set custom acceptable Marker type definitions, ignored if markerStream not None.
:param bool printDebug: If true, prints LSLScanner debug information.
:param GlobalEpochSettings globalEpochSettings: Sets global timing settings for epochs.
:param dict customEpochSettings: Sets individual timing settings for epochs. Dict {marker name string: IndividualEpochSettings()}.
:param dict streamChsDropDict: Keys for dict should be respective datastreams with corresponding list of which channels to drop.
:param dict streamCustomFeatureExtract: Allows dict to be passed of datastream type with custom feature extractor class for analysing data.
:param int minimumEpochsRequired: Minimum number of required epochs before model fitting begins, must be of each type of received markers and more than 1 type of marker to classify.
:param ClassifierMixin clf: Allows custom Sklearn model to be passed.
:param model model: Allows custom tensorflow model to be passed.

.. py:method:: __enter__()

   Connects to the BCI.

.. py:method:: __exit__(exc_type, exc_val, exc_tb)

   Stops all threads of the BCI.

.. py:method:: Connect()

   Checks valid data and markers streams are present, controls dependent functions by setting self.connected.

.. py:method:: TrainMode()

   Set the mode to Train.

.. py:method:: TestMode()

   Set the mode to Test.

.. py:method:: CurrentClassifierInfo()

   Retrieve current classifier info.

.. py:method:: CurrentClassifierMarkerGuess()

   Retrieve current classifier marker guess.

.. py:method:: ReceivedMarkerCount()

   Retrieve received marker count.

.. py:method:: __StartThreads()

   Starts the threads of the BCI.

.. py:method:: StopThreads()

   Stops all threads of the BCI.

.. py:method:: ConfigureMachineLearning(minimumEpochsRequired=10, clf=None, model=None)

   Configure machine learning settings.

   :param int minimumEpochsRequired: Minimum number of epochs required.
   :param ClassifierMixin clf: Allows custom Sklearn model to be passed.
   :param model model: Allows custom tensorflow model to be passed.

.. py:method:: ConfigureEpochWindowSettings(globalEpochSettings=GlobalEpochSettings(), customEpochSettings={})

    Configure epoch window settings.

    :param GlobalEpochSettings globalEpochSettings: Sets global timing settings for epochs.
    :param dict customEpochSettings: Sets individual timing settings for epochs.

.. py:method:: ConfigureDataStreamChannels(streamChsDropDict={})

   Configure data stream channels.

   :param dict streamChsDropDict: Keys for dict should be respective datastreams with corresponding list of which channels to drop.

.. py:method:: ResetThreadsAfterConfigs()

   Reset threads after configurations.
