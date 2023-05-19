Epoch Timing
############

What are Epochs?
----------------
Defining epoch timings relate to the amount of time before and after a marker has been received we wish to process and analyse/extract features from out time series data for training models.

Setting the ``globalEpochSettings`` with the ``GlobalEpochSettings()`` class sets the target window length and overlap for the training time windows.


Setting Custom Epoch Times
------------------------

The figure below illustrates when you may have epochs of differing lengths received on the LSL marker stream. A baseline marker may signify an extended period, in this case 10 seconds, and our motor task is only 1 second long. To account for this we can set our ``customEpochSettings`` and ``globalEpochSettings`` accordingly:

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

.. _overlap0:

.. image:: ../Images/splitEpochs/example1split0%25.png
   :target: https://github.com/LMBooth/pybci/blob/main/docs/Images/splitEpochs/example1split0%25.png
   
By setting ``gs.windowOverlap`` to 0.5 we can turn overlap our 1 second epochs by 50% giving us 19 (2n-1) epochs, shown below:

.. _overlap50:

.. image:: ../Images/splitEpochs/example1split50%25.png
   :target: https://github.com/LMBooth/pybci/blob/main/docs/Images/splitEpochs/example1split50%25.png
