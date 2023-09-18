Pseudo Device
############

What is the Pseudo Device?
=========================================================
For ease of use the variable bool :code:`createPseudoDevice` can be set to True when instantiating :code:`PyBCI()` so the default PseudoDevice is run in another process enabling examples to be run without the need of LSL enabled hardware.

The PseudoDevice class and PseudoDeviceController can be used when the user has no available LSL marker or data streams, allowing for quick and simple execution of the examples. The Pseudo Device enables testing pipelines without the need of configuring and running LSL enabled hardware.

The PseudoDevice class holds marker information and generates signal data based on the given configuration set in :ref:`configuring-pseudo-device`.

The PseudoDeviceController can have the string "process" or "thread" set to decide whether the pseudo device should be a multiprocessed or threaded operation respectively, by default it is set to "process", then passes all the same configuration arguments to PseudoDevice.

Any generic LSLViewer can be used to view the generated data, `example viewers found on this link. <https://labstreaminglayer.readthedocs.io/info/viewers.html>`_.

.. _configuring-pseudo-device:
Configuring the Pseudo Device
=========================================================

By default the PseudoDevice has 4 markers, "baseline", "Marker1", "Marker2", "Marker3" and "Marker4", each with peak frequencies of 3, 8, 10 and 12 Hz respectively.
Each signal is modified for 1 second after the marker has occurred, and the seconds between the markers are spaced by 5 seconds.
  
Upon creating PyBCI object a dict of the following kwargs can be passed to dictate the behaviour of the pseudo device:

.. code-block::

  stop_signal – multiprocessing.Event or bool: Signal used to stop the device’s operation.
  is_multiprocessing – bool: Flag indicating if the class instance is running in a multiprocessing environment. Default is True.
  markerConfigStrings – list(str): List of marker strings used for generating marker data. Default is [“Marker1”, “Marker2”, “Marker3”].
  pseudoMarkerDataConfigs – list: List of PseudoDataConfig objects for configuring the pseudo EMG signals. If None, default configurations will be used.
  pseudoMarkerConfig – PseudoMarkerConfig: Configuration settings for pseudo markers. Default is PseudoMarkerConfig.
  dataStreamName – string: Name to be assigned to the data stream. Default is “PyBCIPseudoDataStream”.
  dataStreamType – string: Type of the data stream (e.g., “EMG”). Default is “EMG”.
  sampleRate – int: The sample rate in Hz for the data stream. Default is 250.
  channelCount – int: The number of channels for the data stream. Default is 8.
  logger – Logger: Logger object for logging activities. Default is Logger(Logger.INFO).
  log_queue – multiprocessing.Queue: Queue object for logging activities in a multiprocessing environment. Default is None.


Where PseudoDataConfig and PseudoDataConfig are:

.. code-block:: python

  class PseudoDataConfig:
      duration = 1.0 
      noise_level = 1
      amplitude = 2
      frequency = 3
  
  class PseudoMarkerConfig:
      markerName = "PyBCIPseudoMarkers"
      markerType = "Markers"
      baselineMarkerString = "baseline"
      repeat = True
      autoplay = True
      num_baseline_markers = 10
      number_marker_iterations = 10
      seconds_between_markers = 5
      seconds_between_baseline_marker = 10
      baselineConfig = PseudoDataConfig()

Two LSL streams are then created, one marker stream for informing pybci an event has occurred and a datastream which has the corresponding altered data to train the applied model with. 
