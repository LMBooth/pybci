Pseudo Device
############

What is the Pseudo Device?
=========================================================
For ease of use the variable bool :code:`createPseudoDevice` can be set to True when instantiating :code:`PyBCI()` so the default PseudoDevice is run in another process enabling examples to be run without the need of LSL enabled hardware.

The PseudoDevice class and PseudoDeviceController can be used when the user has no available LSL marker or data streams, allowing for quick and simple execution of the examples. The Pseudo Device enables testing pipelines without the need of configuring and running LSL enabled hardware.

The PseudoDevice class holds marker information and generates signal data based on the given configuration set in :ref:`configuring-pseudo-device`.

The PseudoDeviceController can have the string "process" or "thread" set to decide whether the pseudo device should be a multiprocessed or threaded operation respectively, by default it is set to "process", then passes all the same configuration arguments to PseudoDevice.

.. _configuring-pseudo-device:
Configuring the Pseudo Device
=========================================================

By default the PseudoDevice has 4 markers, "baseline", "Marker1", "Marker2", "Marker3" and "Marker4", each with peak frequencies of 3, 8, 10 and 12 Hz respectively.
Each signal is modified for 1 second after the marker has occurred, and the seconds between the markers are spaced by 5 seconds.

Upon creating the PseudoDevice

Data Thread
**********************************************

Marker Thread
**********************************************
