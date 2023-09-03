pybci.Utils.PseudoDevice import PseudoDevice
============

.. class:: PseudoDevice(stop_signal, is_multiprocessing=True, markerConfigStrings=["Marker1", "Marker2", "Marker3"], pseudoMarkerDataConfigs=None, pseudoMarkerConfig=PseudoMarkerConfig, dataStreamName="PyBCIPseudoDataStream", dataStreamType="EMG", sampleRate=250, channelCount=8, logger=Logger(Logger.INFO), log_queue=None)

   The PseudoDevice class is responsible for generating pseudo EMG signals and markers. It is designed to either run in a multiprocessing environment or as a standalone thread, depending on the ``is_multiprocessing`` parameter.

   :param stop_signal: multiprocessing.Event or bool: Signal used to stop the device's operation.
   :param is_multiprocessing: bool: Flag indicating if the class instance is running in a multiprocessing environment. Default is `True`.
   :param markerConfigStrings: list(str): List of marker strings used for generating marker data. Default is ["Marker1", "Marker2", "Marker3"].
   :param pseudoMarkerDataConfigs: list: List of `PseudoDataConfig` objects for configuring the pseudo EMG signals. If `None`, default configurations will be used.
   :param pseudoMarkerConfig: PseudoMarkerConfig: Configuration settings for pseudo markers. Default is `PseudoMarkerConfig`.
   :param dataStreamName: string: Name to be assigned to the data stream. Default is "PyBCIPseudoDataStream".
   :param dataStreamType: string: Type of the data stream (e.g., "EMG"). Default is "EMG".
   :param sampleRate: int: The sample rate in Hz for the data stream. Default is 250.
   :param channelCount: int: The number of channels for the data stream. Default is 8.
   :param logger: Logger: Logger object for logging activities. Default is `Logger(Logger.INFO)`.
   :param log_queue: multiprocessing.Queue: Queue object for logging activities in a multiprocessing environment. Default is `None`.

   .. note::
      Ensure that the `stop_signal` is set appropriately, especially if running in a multiprocessing environment. 

   .. py:method:: log_message(level='INFO', message="")

      Logs a message either to the `log_queue` or to the logger object, depending on the mode of operation.

      :param level: string: The level of the log message (e.g., "INFO", "ERROR"). Default is "INFO".
      :param message: string: The message to be logged.

   .. py:method:: GeneratePseudoEMG(samplingRate, duration, noise_level, amplitude, frequency)

      Generates a pseudo EMG signal based on the provided parameters. This method uses a sinusoidal model to create the EMG signal and adds Gaussian noise to it.

      :param samplingRate: int: The sampling rate in Hz.
      :param duration: float: The duration of the signal in seconds.
      :param noise_level: float: The amplitude of the Gaussian noise to be added.
      :param amplitude: float: The amplitude of the sinusoidal EMG signal.
      :param frequency: float: The frequency of the sinusoidal EMG signal in Hz.

   .. py:method:: update()

      Updates the internal state of the PseudoDevice, including data and marker generation. This method should be called in a loop for continuous operation.

   .. py:method:: StopStreaming()

      Stops the data and marker streaming. This method sets the `stop_signal` to terminate the streaming.

      .. note::
         It is recommended to call this method for a graceful shutdown.

   .. py:method:: BeginStreaming()

      Initiates the streaming of pseudo EMG data and markers based on the provided configurations. This method should be called to start the device's operation.

   .. py:method:: StartMarkers()

      Initiates the generation and streaming of markers based on the `markerConfigStrings` and `pseudoMarkerConfig` settings. This method should be called to start marker streaming.

