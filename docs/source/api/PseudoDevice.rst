pybci.Utils.PseudoDevice import PseudoDeviceController
========================

.. class:: PseudoDeviceController(is_multiprocessing=True, markerConfigStrings=["Marker1", "Marker2", "Marker3"], pseudoMarkerDataConfigs=None, createMarkers=True, pseudoMarkerConfig=PseudoMarkerConfig, dataStreamName="PyBCIPseudoDataStream", dataStreamType="EMG", sampleRate=250, channelCount=8, logger=Logger(Logger.INFO), log_queue=None)

   The PseudoDeviceController class is designed for generating pseudo EMG signals and markers, simulating a Lab Streaming Layer (LSL) device. It supports both multiprocessing and threading environments, depending on the `is_multiprocessing` parameter.

   :param is_multiprocessing: bool: Indicates if the class instance operates in a multiprocessing environment. Default is `True`.
   :param markerConfigStrings: list(str): Marker strings for generating marker data. Default is ["Marker1", "Marker2", "Marker3"].
   :param pseudoMarkerDataConfigs: list: Configurations for pseudo EMG signals. Uses default configurations if `None`.
   :param createMarkers: bool: Flag to determine if markers should be created. Default is `True`.
   :param pseudoMarkerConfig: PseudoMarkerConfig: Settings for pseudo markers.
   :param dataStreamName: string: Name for the data stream.
   :param dataStreamType: string: Data stream type (e.g., "EMG").
   :param sampleRate: int: Sample rate in Hz.
   :param channelCount: int: Number of channels.
   :param logger: Logger: Logger object for logging.
   :param log_queue: multiprocessing.Queue or queue.Queue: Queue object for logging.

   .. note::
      The sample rate is not exact and may vary with CPU strain. 

   .. py:method:: BeginStreaming()

      Initiates streaming of pseudo EMG data and markers. This method should be called to start the device's operation.

   .. py:method:: StopStreaming()

      Stops the data and marker streaming, signaling the termination.

   .. py:method:: log_message(level='INFO', message="")

      Logs a message to the `log_queue` or directly, based on the operation mode.

      :param level: string: Log message level (e.g., "INFO", "ERROR").
      :param message: string: Message to log.

   .. note::
      Ensure a graceful shutdown by calling `StopStreaming()`.

.. note::
   The PseudoDeviceController is suitable for simulations and testing purposes. It may require specific setup for multiprocessing or threading environments.
