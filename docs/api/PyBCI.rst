PyBCI
=====

.. class:: PyBCI(dataStreams=None, markerStream=None, streamTypes=None, markerTypes=None, loggingLevel=Logger.INFO, globalEpochSettings=GlobalEpochSettings(), customEpochSettings={}, streamChsDropDict={}, streamCustomFeatureExtract={}, minimumEpochsRequired=10, clf=None, model=None, torchModel=None)

   The `PyBCI` object is the main controller for interfacing with all relevant threads. When initialised sets up the main operation of the BCI and can be queried for relevnt information

   **Parameters:**

.. _pybci-datastreams:

.. py:attribute:: dataStreams
    :type: list(str) or None
    :value: Allows the user to set custom acceptable EEG stream definitions. If `None`, it defaults to `streamTypes` scan.

.. _pybci-markerstream:

.. py:attribute:: markerStream
    :type: list(str) or None
    :value: Allows the user to set custom acceptable Marker stream definitions. If `None`, it defaults to `markerTypes` scan.

.. _pybci-streamtypes:

.. py:attribute:: streamTypes
    :type: list(str) or None
    :value: Allows the user to set custom acceptable EEG type definitions, ignored if `dataStreams` is not `None`.

.. _pybci-markertypes:

.. py:attribute:: markerTypes
    :type: list(str) or None
    :value: Allows the user to set custom acceptable Marker type definitions, ignored if `markerStream` is not `None`.

.. _pybci-logginglevel:

.. py:attribute:: loggingLevel
    :type: string
    :value: Sets PyBCI print level, ('INFO' prints all statements, 'WARNING' is only warning messages, 'TIMING' gives estimated time for feature extraction, and classifier training or testing, 'NONE' means no prints from PyBCI).

.. _pybci-globalepochsettings:

.. py:attribute:: globalEpochSettings
    :type: GlobalEpochSettings
    :value: Sets global timing settings for epochs. See :ref:`set_custom_epoch_times`.                                                                                    
                                                 

.. _pybci-customepochsettings:

.. py:attribute:: customEpochSettings
    :type: dict
    :value: Sets individual timing settings for epochs. {markerstring1:IndividualEpochSettings(),markerstring2:IndividualEpochSettings()}

.. _pybci-streamchsdropdict:

.. py:attribute:: streamChsDropDict
    :type: dict
    :value: Keys for dict should be respective datastreams with corresponding list of which channels to drop. {datastreamstring1: list(ints), datastreamstring2: list(ints)}

.. _pybci-streamcustomfeatureextract:

.. py:attribute:: streamCustomFeatureExtract
    :type: dict
    :value: Allows dict to be passed of datastream with custom feature extractor class for analyzing data. {datastreamstring1: customClass1(), datastreamstring2: customClass1()}

.. _pybci-minimumepochsrequired:

.. py:attribute:: minimumEpochsRequired
    :type: int
    :value: Minimum number of required epochs before model fitting begins, must be of each type of received markers and more than 1 type of marker to classify.

.. _pybci-clf:

.. py:attribute:: clf
    :type: sklearn.base.ClassifierMixin or None
    :value: Allows custom Sklearn model to be passed.                                                                          
                                                          

.. _pybci-model:

.. py:attribute:: model
    :type: tf.keras.model or None
    :value: Allows custom TensorFlow model to be passed.                                                                                                                                    


.. _pybci-torchmodel:

.. py:attribute:: torchModel
    :type: custom def or None
    :value: Custom torch function should be passed with 4 inputs (x_train, x_test, y_train, y_test). Needs to return [accuracy, model], look at testPyTorch.py in examples for reference.

.. _pybci-enter:

.. py:method:: __enter__()

   Connects to the BCI.

.. _pybci-exit:

.. py:method:: __exit__(exc_type, exc_val, exc_tb)

   Stops all threads of the BCI.

.. _pybci-connect:

.. py:method:: Connect()

   Checks if valid data and marker streams are present, controls dependent functions by setting self.connected. Returns a boolean indicating the connection status.

.. py:method:: TrainMode()

   Set the mode to Train. The BCI will try to connect if it is not already connected.

.. py:amethod:: TestMode()

   Set the mode to Test. The BCI will try to connect if it is not already connected.

.. py:method:: CurrentClassifierInfo()

   :returns: a dictionary containing "clf", "model," "torchModel," and "accuracy." The accuracy is 0 if no model training/fitting has occurred. If the mode is not used, the corresponding value is None. If not connected, returns `{"Not Connected": None}`.

.. py:method:: CurrentClassifierMarkerGuess()

   :returns: an integer or None. The returned integer corresponds to the value of the key from the dictionary obtained from `ReceivedMarkerCount()` when in test mode. If in train mode, returns None.

.. py:method:: CurrentFeaturesTargets()

   :returns: a dictionary containing "features" and "targets." "features" is a 2D list of feature data, and "targets" is a 1D list of epoch targets as integers. If not connected, returns `{"Not Connected": None}`.

.. py:method:: ReceivedMarkerCount()

   :returns: a dictionary. Each key is a string received on the selected LSL marker stream, and the value is a list. The first item is the marker id value, to be used with `CurrentClassifierMarkerGuess()`. The second value is a received count for that marker type. Will be empty if no markers are received.
