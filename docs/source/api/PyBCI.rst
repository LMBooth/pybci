PyBCI
=====

.. class:: PyBCI(dataStreams=None, markerStream=None, streamTypes=None, markerTypes=None, loggingLevel=Logger.INFO, globalEpochSettings=Global EpochSettings(), custom EpochSettings={}, streamChsDropDict={}, streamCustomFeatureExtract={}, minimum EpochsRequired=10, createPseudoDevice=False, pseudoDeviceArgs=None, clf=None, model=None, torchModel=None)

   The PyBCI object is the main controller for interfacing with all relevant threads. When initialised, it sets up the main operation of the BCI and can be queried for relevant information.

   :param dataStreams: list(str) or None: Allows the user to set custom acceptable EEG stream definitions. Defaults to `streamTypes` scan if `None`.
   :param markerStream: str or None: Allows the user to set custom acceptable Marker stream definition. Defaults to `markerTypes` scan if `None`.
   :param streamTypes: list(str) or None: Allows the user to set custom acceptable EEG type definitions, ignored if `dataStreams` is not `None`.
   :param markerTypes: list(str) or None: Allows the user to set custom acceptable Marker type definitions, ignored if `markerStream` is not `None`.
   :param loggingLevel: string: Sets PyBCI print level. Options are 'INFO', 'WARNING', 'TIMING', and 'NONE'.
   :param global EpochSettings: Global EpochSettings: Sets global timing settings for epochs.
   :param custom EpochSettings: dict: Sets individual timing settings for epochs.
   :param streamChsDropDict: dict: Specifies which channels to drop for each data stream.
   :param streamCustomFeatureExtract: dict: Allows a custom feature extractor class for each data stream.
   :param minimum EpochsRequired: int: Minimum number of required epochs before model fitting begins.
   :param createPseudoDevice: bool: If True, auto-generates LSL marker and LSL data.
   :param pseudoDeviceArgs: dict: Dictionary of arguments to initialize pseudo device.
   :param clf: sklearn.base.ClassifierMixin or None: Allows custom Sklearn model to be passed.
   :param model: tf.keras.model or None: Allows custom TensorFlow model to be passed.
   :param torchModel: custom def or None: Custom torch function should be passed with 4 inputs.

   .. note::
      For more information on epoch settings, see `Global EpochSettings()` and `Individual EpochSettings()`.

   .. py:method:: __enter__()

      Connects to the BCI. Same as __init__.

   .. py:method:: __exit__(exc_type, exc_val, exc_tb)

      Stops all threads of the BCI.

   .. py:method:: Connect()

      Checks for valid data and marker streams, sets `self.connected`. Returns boolean indicating connection status.

   .. py:method:: TrainMode()

      Sets mode to Train. Tries to connect if not already connected.

   .. py:method:: TestMode()

      Sets mode to Test. Tries to connect if not already connected.

   .. py:method:: CurrentClassifierInfo()

      :returns: Dictionary containing "clf", "model," "torchModel," and "accuracy." If not connected, returns `{"Not Connected": None}`.

   .. py:method:: CurrentClassifierMarkerGuess()

      :returns: Integer or None. Returns integer corresponding to value of key from `ReceivedMarkerCount()` dictionary. Returns None if in Train mode.

   .. py:method:: CurrentFeaturesTargets()

      :returns: Dictionary containing "features" and "targets." If not connected, returns `{"Not Connected": None}`.

   .. py:method:: ReceivedMarkerCount()

      :returns: Dictionary where each key is a received marker string and the value is a list. The list contains the marker ID and received count for that marker type.
