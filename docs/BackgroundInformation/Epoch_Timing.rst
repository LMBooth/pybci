.. _epoch_timing:
Epoch Timing
############

What are Epochs?
----------------
Epochs are periods of time.

In relation to training models on set actions for brain computer interfaces, it is useful to define epochs by the amount of time-series data before and after a marker has been received indicating a desired classification action has been executed. A selected LSL marker stream is used to send strings which represent unique target training markers. Once data is sliced in to epochs it may be processed for feature extraction, or in some cases passed as raw time-series input, see :ref:`custom-extractor` and :ref:`raw-extractor` for more information on feature extraction.

Setting the :py:data:`globalEpochSettings` with the :class:`GlobalEpochSettings()` class sets the target window length and overlap for the training time windows. It is desirable to have a single global window length that all epochs are sliced to match, this gives a uniform array when passing to the classifier. When in testing mode a having a continuous rolling window of data is sliced to this size and overlapped based on the windowOverlap, see :ref:`set_custom_epoch_times` for more info.

.. _set_custom_epoch_times:
Setting Custom Epoch Times
------------------------

The figure below illustrates when you may have epochs of differing lengths received on the LSL marker stream. A baseline marker may signify an extended period, in this case 10 seconds, and our motor task is only 1 second long. To account for this set :py:data:`customEpochSettings` and :py:data:`globalEpochSettings` accordingly:

.. code-block:: python

   gs = GlobalEpochSettings()
   gs.tmax = 1 # grab 1 second after marker
   gs.tmin = 0 # grabs 0 seconds before marker
   gs.splitCheck = False # splits samples between tmin and tmax
   gs.windowLength = 1 # window length of 1 s
   gs.windowOverlap = 0.5 # windows overap by 50%, so for a total len
   baselineSettings = IndividualEpochSetting()
   baselineSettings.splitCheck = False
   baselineSettings.tmin = 0      # time in seconds to capture samples before trigger
   baselineSettings.tmax=  10      # time in seconds to capture samples after trigger
   bci = PyBCI(customEpochSettings=baselineSettings, globalEpochSettings=gs)

Highlighting these epochs on some psuedo emg data looks like the following:

.. _nosplitExample:

.. image:: ../Images/splitEpochs/example1.png
   :target: https://github.com/LMBooth/pybci/blob/main/docs/Images/splitEpochs/example1.png


Overlapping Epoch Windows
------------------------

By setting splitCheck to True for ``baselineSettings.splitCheck`` and ``gs.windowOverlap`` to 0 we can turn one marker into 10 epochs, shown below:

.. _nooverlapExample:

.. image:: ../Images/splitEpochs/example1split0.png
   :target: https://github.com/LMBooth/pybci/blob/main/docs/Images/splitEpochs/example1split0.png
   
   
By setting ``gs.windowOverlap`` to 0.5 we can turn overlap our 1 second epochs by 50% giving us 19 (2n-1) epochs, shown below:

.. _overlapExample:

.. image:: ../Images/splitEpochs/example1split50.png
   :target: https://github.com/LMBooth/pybci/blob/main/docs/Images/splitEpochs/example1split50.png
   
   
Debugging Timing Errors
------------------------
When initialising the :class:`PyBCI()` class set :py:data:`loggingLevel` to “TIMING” to time the feature extraction time for each data inlet as well as classification testing and training times. These are the most computationally intensive tasks and will induce the most lag in the the system. Each printed time must be shorter then :py:data:`globalEpochSettings`.windowLength*(1- :py:data:`globalEpochSettings`.windowOverlap) to minimise delays from input data action to classification output.
