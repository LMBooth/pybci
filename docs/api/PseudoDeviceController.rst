pybci.Configuration.PseudoDevice import PseudoDeviceController
======================

.. class:: PseudoDeviceController(execution_mode='process', *args, **kwargs)

   The PseudoDeviceController class serves as a manager for the PseudoDevice. It is responsible for controlling the behavior of the PseudoDevice, including starting and stopping data streams. The class is designed to run either in multiprocessing mode or threading mode, depending on the ``execution_mode`` parameter.

   :param execution_mode: string: Determines the mode of execution. It can be either 'process' for multiprocessing or 'thread' for multi-threading. Default is 'process'.
   :param args: tuple: Additional positional arguments that are passed directly to the PseudoDevice instance.
   :param kwargs: dict: Additional keyword arguments that are passed directly to the PseudoDevice instance.

   .. note:: 
      The `execution_mode` parameter is crucial for defining the type of parallelization to use. Ensure to set it appropriately based on your application requirements.

   .. py:method:: _run_device()

      This is an internal method used to initiate the execution of the PseudoDevice. It runs the device based on the execution mode specified during the initialization.

      :raises ValueError: If an unsupported execution mode is provided.
      
      .. warning::
         This method is intended for internal use only and should not be called directly.

   .. py:method:: _should_stop()

      An internal method that checks whether the execution of the PseudoDevice should be stopped based on the ``stop_signal``.

      :returns: `True` if the execution should be stopped, `False` otherwise.

      .. warning::
         This method is intended for internal use only and should not be called directly.

   .. py:method:: BeginStreaming()

      Starts the streaming of pseudo EMG data and markers. This method initiates the generation and transmission of data based on the configurations provided.

      .. note:: 
         The actual behavior of the data streaming is defined in the PseudoDevice class. This method serves as a controller interface.

   .. py:method:: StopStreaming()

      Stops the data streaming process. This method sets the ``stop_signal`` flag, which subsequently terminates the data generation and transmission.

      .. note:: 
         It is recommended to call this method for graceful termination of the device.

   .. py:method:: close()

      This method is responsible for cleaning up resources and ensuring a graceful shutdown of the controller. It joins the worker threads or processes and performs necessary cleanup operations.

      .. note::
         Always call this method before terminating the application to ensure proper resource management.

