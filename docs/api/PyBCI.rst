PyBCI
=====

.. class:: PyBCI(dataStreams=None, markerStream=None, streamTypes=None, markerTypes=None, loggingLevel=Logger.INFO, globalEpochSettings=GlobalEpochSettings(), customEpochSettings={}, streamChsDropDict={}, streamCustomFeatureExtract={}, minimumEpochsRequired=10, clf=None, model=None, torchModel=None)

   The `PyBCI` object stores data from available LSL time series data streams (EEG, pupilometry, EMG, etc.) and holds a configurable number of samples based on LSL marker strings.

   **Parameters:**

.. py:attribute:: dataStreams
:dataStreams: list(str) or None: Allows the user to set custom acceptable EEG stream definitions. If `None`, it defaults to `streamTypes` scan.
.. py:attribute:: markerStream
:markerStream: list(str) or None: Allows the user to set custom acceptable Marker stream definitions. If `None`, it defaults to `markerTypes` scan.
.. py:attribute:: streamTypes
:streamTypes: list(str) or None: Allows the user to set custom acceptable EEG type definitions, ignored if `dataStreams` is not `None`.
.. py:attribute:: markerTypes
:markerTypes: list(str) or None: Allows the user to set custom acceptable Marker type definitions, ignored if `markerStream` is not `None`.
.. py:attribute:: loggingLevel
:loggingLevel: string: Sets PyBCI print level, ('INFO' prints all statements, 'WARNING' is only warning messages, 'TIMING' gives estimated time for feature extraction, and classifier training or testing, 'NONE' means no prints from PyBCI)
.. py:attribute:: globalEpochSettings
:globalEpochSettings: GlobalEpochSettings: Sets global timing settings for epochs. 
.. py:attribute:: customEpochSettings
:customEpochSettings: dict: Sets individual timing settings for epochs. {markerstring1:IndividualEpochSettings(),markerstring2:IndividualEpochSettings()}
.. py:attribute:: streamChsDropDict
:streamChsDropDict: dict: Keys for dict should be respective datastreams with corresponding list of which channels to drop. {datastreamstring1: list(ints), datastreamstring2: list(ints)}
.. py:attribute:: streamCustomFeatureExtract
:streamCustomFeatureExtract: dict::streamCustomFeatureExtract: Allows dict to be passed of datastream with custom feature extractor class for analyzing data. {datastreamstring1: customClass1(), datastreamstring2: customClass1()}
.. py:attribute:: minimumEpochsRequired
:type minimumEpochsRequired: int: Minimum number of required epochs before model fitting begins, must be of each type of received markers and more than 1 type of marker to classify.
.. py:attribute:: clf
:type clf: sklearn.base.ClassifierMixin or None: Allows custom Sklearn model to be passed.
.. py:attribute:: model
:type model: tf.keras.model or None: Allows custom TensorFlow model to be passed.
.. py:attribute:: torchModel
:type torchModel: custom def or None: Custom torch function should be passed with 4 inputs (x_train, x_test, y_train, y_test). Needs to return [accuracy, model], look at testPyTorch.py in examples for reference.

.. py:method:: __enter__()

   Connects to the BCI.

.. py:method:: __exit__(exc_type, exc_val, exc_tb)

   Stops all threads of the BCI.

.. py:method:: Connect()

   Checks if valid data and marker streams are present, controls dependent functions by setting self.connected. Returns a boolean indicating the connection status.

.. py:method:: TrainMode()

   Set the mode to Train. The BCI will try to connect if it is not already connected.

.. py:method:: TestMode()

   Set the mode to Test. The BCI will try to connect if it is not already connected.

.. py:method:: CurrentClassifierInfo()

   :returns: a dictionary containing "clf", "model," "torchModel," and "accuracy." The accuracy is 0 if no model training/fitting has occurred. If the mode is not used, the corresponding value is None. If not connected, returns `{"Not Connected": None}`.

.. py:method:: CurrentClassifierMarkerGuess()

   :returns: an integer or None. The returned integer corresponds to the value of the key from the dictionary obtained from `ReceivedMarkerCount()` when in test mode. If in train mode, returns None.

.. py:method:: CurrentFeaturesTargets()

   :returns: a dictionary containing "features" and "targets." "features" is a 2D list of feature data, and "targets" is a 1D list of epoch targets as integers. If not connected, returns `{"Not Connected": None}`.

.. py:method:: ReceivedMarkerCount()

   :returns: a dictionary. Each key is a string received on the selected LSL marker stream, and the value is a list. The first item is the marker id value, to be used with `CurrentClassifierMarkerGuess()`. The second value is a received count for that marker type. Will be empty if no markers are received.
