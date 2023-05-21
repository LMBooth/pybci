PyBCI
=====
.. class:: PyBCI(dataStreams=None, markerStream=None, streamTypes=None, markerTypes=None, printDebug=True, globalEpochSettings=GlobalEpochSettings(), customEpochSettings={}, streamChsDropDict={}, streamCustomFeatureExtract={}, minimumEpochsRequired=10, clf=None, model=None)

    The `PyBCI` object stores data from available LSL time series data streams (EEG, pupilometry, EMG, etc.) and holds a configurable number of samples based on LSL marker strings.

    **Parameters:**

    :dataStreams: list(str) or None
        Allows the user to set custom acceptable EEG stream definitions. If `None`, it defaults to `streamTypes` scan.

    :markerStream: list(str) or None
        Allows the user to set custom acceptable Marker stream definitions. If `None`, it defaults to `markerTypes` scan.

    :streamTypes: list(str) or None
        Allows the user to set custom acceptable EEG type definitions, ignored if `dataStreams` is not `None`.

    :markerTypes: list(str) or None
        Allows the user to set custom acceptable Marker type definitions, ignored if `markerStream` is not `None`.

    :printDebug: bool
        If `True`, prints LSLScanner debug information.

    :globalEpochSettings: GlobalEpochSettings
        Sets global timing settings for epochs.

    :customEpochSettings: dict{str: IndividualEpochSettings}
        Sets individual timing settings for epochs. The keys of the dictionary are marker name strings, and the values are `IndividualEpochSettings` objects.

    :streamChsDropDict: dict{str: list(int)}
        Keys for dict should be respective datastreams with corresponding lists of which channels to drop.

    :streamCustomFeatureExtract: dict{str:class}
        Allows a dictionary to be passed with datastream type as the key and a custom feature extractor class for analyzing data as the value.

    :minimumEpochsRequired: int
        Minimum number of required epochs before model fitting begins, must be of each type of received markers and more than 1 type of marker to classify.

    :clf: ClassifierMixin or None
        Allows a custom Sklearn model to be passed, if None and None model too defaults to sklearn SVM with 'rbf' kernel.

    :model: model or None
        Allows a custom TensorFlow model to be passed, if None and None model too defaults to sklearn SVM with 'rbf' kernel.

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

   Retrieve current classifier info. Give dict of current fit model, clf and the class accuracy, if sklearn is used model is None, if tensorflow is used clf is None. the clf or model is fit when the minimum number of training epochs have been recied for each marker, default 10.
   
   :returns: 
        dict{
            "clf": clf,
            "model": model,
            "accuracy": accuracy
        }

.. py:method:: CurrentClassifierMarkerGuess()

   Retrieve current classifier marker guess.
   
   :returns: `int` if in test mode, `None` is in train mode. The int should relate to the dict value from :method:`ReceivedMarkerCount()`

.. py:method:: ReceivedMarkerCount()

   Retrieve received marker count.

    :returns: dict{str:[int,int]}, where the string is the marker label receied on the LSL, the first int is the corresponding value returned by :method:`CurrentClassifierMarkerGuess()`, and the second int is the number of received markers of that key type.

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
