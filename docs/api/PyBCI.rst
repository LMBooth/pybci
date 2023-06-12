PyBCI
=====
.. class:: PyBCI(dataStreams=None, markerStream=None, streamTypes=None, markerTypes=None, loggingLevel=Logger.INFO, globalEpochSettings=GlobalEpochSettings(), customEpochSettings={}, streamChsDropDict={}, streamCustomFeatureExtract={}, minimumEpochsRequired=10, clf=None, model=None, torchModel=None)

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

    :loggingLevel: string
        Sets PyBCI print level, ('INFO' prints all statements, 'WARNING' is only warning messages, 'TIMING' gives estimated time for feature extraction, and classifier training or testing, 'NONE' means no prints from PyBCI)

    :globalEpochSettings: GlobalEpochSettings
        Sets global timing settings for epochs.

    :customEpochSettings: dict
        Sets individual timing settings for epochs. {markerstring1:IndividualEpochSettings(),markerstring2:IndividualEpochSettings()}

    :streamChsDropDict: dict
        Keys for dict should be respective datastreams with corresponding list of which channels to drop. {datastreamstring1: list(ints), datastreamstring2: list(ints)}

    :streamCustomFeatureExtract: dict
        Allows dict to be passed of datastream with custom feature extractor class for analysing data. {datastreamstring1: customClass1(), datastreamstring2: customClass1(),}

    :minimumEpochsRequired: int
        Minimum number of required epochs before model fitting begins, must be of each type of received markers and more than 1 type of marker to classify.

    :clf: sklearn.base.ClassifierMixin or None
        Allows custom Sklearn model to be passed.

    :model: tf.keras.model or None
        Allows custom tensorflow model to be passed.

    :torchModel: [torchModel(), torch.nn.Module] or None
        Currently a list where first item is torchmodel analysis function, second is torch model, check pytorch example - likely to change in future updates.

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

   Retrieve current classifier information including clf, model, torchModel and accuracy.quote("class PyBCI:\n    globalEpochSettings =", "likely to change in future updates")

