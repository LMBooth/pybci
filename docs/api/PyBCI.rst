PyBCI
=====
.. class:: PyBCI(dataStreams=None, markerStream=None, streamTypes=None, markerTypes=None, loggingLevel=Logger.INFO, globalEpochSettings=GlobalEpochSettings(), customEpochSettings={}, streamChsDropDict={}, streamCustomFeatureExtract={}, minimumEpochsRequired=10, clf=None, model=None, torchModel=None)

    The `PyBCI` object stores data from available LSL time series data streams (EEG, pupilometry, EMG, etc.) and holds a configurable number of samples based on LSL marker strings.

    **Parameters:**

    .. pybci:datastreams::
    :dataStreams: list(str) or None
        Allows the user to set custom acceptable EEG stream definitions. If `None`, it defaults to `streamTypes` scan.

    .. pybci:markerStream::
    :markerStream: list(str) or None
        Allows the user to set custom acceptable Marker stream definitions. If `None`, it defaults to `markerTypes` scan.

    .. pybci:streamTypes::
    :streamTypes: list(str) or None
        Allows the user to set custom acceptable EEG type definitions, ignored if `dataStreams` is not `None`.

    .. pybci:markerTypes::
    :markerTypes: list(str) or None
        Allows the user to set custom acceptable Marker type definitions, ignored if `markerStream` is not `None`.

    .. pybci:loggingLevel::
    :loggingLevel: string
        Sets PyBCI print level, ('INFO' prints all statements, 'WARNING' is only warning messages, 'TIMING' gives estimated time for feature extraction, and classifier training or testing, 'NONE' means no prints from PyBCI)

    .. pybci:globalEpochSettings::
    :globalEpochSettings: GlobalEpochSettings
        Sets global timing settings for epochs.

    .. pybci:customEpochSettings::
    :customEpochSettings: dict
        Sets individual timing settings for epochs. {markerstring1:IndividualEpochSettings(),markerstring2:IndividualEpochSettings()}

    .. pybci:streamChsDropDict::
    :streamChsDropDict: dict
        Keys for dict should be respective datastreams with corresponding list of which channels to drop. {datastreamstring1: list(ints), datastreamstring2: list(ints)}

    .. pybci:streamCustomFeatureExtract::
    :streamCustomFeatureExtract: dict
        Allows dict to be passed of datastream with custom feature extractor class for analysing data. {datastreamstring1: customClass1(), datastreamstring2: customClass1(),}

    .. pybci:minimumEpochsRequired::
    :minimumEpochsRequired: int
        Minimum number of required epochs before model fitting begins, must be of each type of received markers and more than 1 type of marker to classify.

    .. pybci:clf::
    :clf: sklearn.base.ClassifierMixin or None
        Allows custom Sklearn model to be passed.

    .. pybci:model::
    :model: tf.keras.model or None
        Allows custom tensorflow model to be passed.

    .. pybci:torchModel::
    :torchModel: custom def or None
        Custom torch function should be passed with 4 inputs (x_train, x_test, y_train, y_test). Needs to return [accuracy, model], look at testPyTorch.py in examples for reference.

.. py:method:: __enter__()

   Connects to the BCI.

.. py:method:: __exit__(exc_type, exc_val, exc_tb)

   Stops all threads of the BCI.

.. py:method:: Connect()

   Checks valid data and markers streams are present, controls dependent functions by setting self.connected. Returns a boolean indicating the connection status.

.. py:method:: TrainMode()

   Set the mode to Train. The BCI will try to connect if it is not already connected.

.. py:method:: TestMode()

   Set the mode to Test. The BCI will try to connect if it is not already connected.

.. py:method:: CurrentClassifierInfo()

   Returns dict. 
        dict of "clf", "model, "torchModel"" and "accuracy" where accuracy is 0 if no model training/fitting has occurred. If mode not used corresponding value is None. If not connected returns {"Not Connected": None}

.. py:method:: CurrentClassifierMarkerGuess()

   Returns int | None. 
        Returned int correlates to value of key from dict from ReceivedMarkerCount() when in testmode. If in trainmode returns None.

.. py:method:: CurrentFeaturesTargets()

    Returns dict. 
        dict of "features" and "targets" where features is 2d list of feature data and targets is a 1d list of epoch targets as ints. If not connected returns {"Not Connected": None}

.. py:method:: ReceivedMarkerCount()

    Returns dict. 
        Every key is a string received on the selected LSL marker stream, the value is a list where the first item is the marker id value, use with CurrentClassifierMarkerGuess() the second value is a received count for that marker type. Will be empty if no markers received.
