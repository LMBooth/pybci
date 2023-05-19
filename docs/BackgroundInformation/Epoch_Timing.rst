Epoch Timing
############

What are Epochs?
----------------
Defining epoch timings relate to the amount of time before and after a marker has been recieved we wish to process and analyse/extract features from out time series data.

By setting the ``globalEpochSettings`` with the ``GlobalEpochSettings()`` class sets the target window length and overlap


Setting Custom Epoch Times
------------------------

The :ref:`figure below <_nosplitExample>` illustrates when you may have epochs of differing lengths received on the LSL marker stream. A baseline marker may signfy an extended period, in this case 10 seconds, and our motor task is only 1 second long.

This would be configured by coding the following :



.. _nosplitExample:

.. image:: Images/splitEpochs/example1.png
   :target: https://github.com/LMBooth/pybci/blob/main/docs/Images/splitEpochs/example1.png


Overlapping Epoch Windows
------------------------


.. _overlap0:

.. image:: Images/splitEpochs/example1split0%25.png
   :target: https://github.com/LMBooth/pybci/blob/main/docs/Images/splitEpochs/example1split0%25.png
   
   
.. _overlap50:

.. image:: Images/splitEpochs/example1split50%25.png
   :target: https://github.com/LMBooth/pybci/blob/main/docs/Images/splitEpochs/example1split50%25.png
