.. _examples:
Examples
############

The following examples can all be found on the `PyBCI github <https://github.com/LMBooth/pybci/tree/main/pybci/Examples>`_.

.. list-table:: PyBCI Examples
   :widths: 25 75
   :header-rows: 1

   * - Example File
     - Description
   * - `ArduinoHandGrasp/ <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/ArduinoHandGrasp>`_
     - Folder contains LSL marker creator with MarkerMaker.py using `PyQt5` as an on screen text stimulus, illustrates how LSL markers can be used to train. `testArduinoHand.py` is tailored for the io:bio (DOI: 10.1109/ACCESS.2021.3079992) device which had its 2nd and 3rd differential channels (channels 21 and 22 out of 24) attached to electrodes on the fore arm with appropriate features extracted. The `testArduinoHand.py` also connected via a serial port to an arduino board which controls serveral servo motors for a motorised hand (see accompanying video), the .ino script for controlling this hand via the arduino can also be `found here <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/ArduinoHandGrasp/ServoControl/ServoControl.ino>`_.
   * - `PsuedoLSLSreamGenerator/mainSend.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PsuedoLSLStreamGenerator/mainSend.py>`_
     - Generates multiple channels on a given stream type at a given sample rate. A baseline signal is generated on an LSL stream outlet and a PyQt button can be pressed to signify this signal on a separate LSL marker stream. The signal can be altered by 5 distinct markers for a configurable amount of time, allowing the user to play with various signal patterns for clasification. NOTE: Requires `PyQt5` and `pyqtgraph` installs for data viewer.
   * - `PupilLabs/bciGazeExample.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/bciGazeExample.py>`_
     - Illustrates how a 'simple' custom pupil-labs feature extractor class can be passed for the gaze data, where the mean pupil diameter is taken for each eye and both eyes and used as feature data, where nans for no confidence are set to a value of 0.
   * - `PupilLabs/RightLeftMarkers.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/RightLeftMarkers.py>`_
     - Uses tkinter to generate visual on-screen stimuli for only right, left or both eyes open, sends same onscreen stimuli as LSL markers, ideal for testing pupil-labs eyes classifier test.
   * - `testEpochTimingsConfig.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testEpochTimingsConfig.py>`_
     - Simple example showing custom global epoch settings  changed on initialisation. Instead of epoching data from 0 to 1 second after the marker we take it from 0.5 seconds before to 0.5 seconds after the marker. 
   * - `testMultimodal.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testMultimodal.py>`_ 
     - Advanced example illustrating two devices, pupil labs gaze device stream wth custom feature extractor class and Hull University ioBio EEG device with the generic feature extractor, each have set channels dropped to reduce computational strain (Async datathreads {LSLsample rate of 0Hz} can be heavy with lots of channels.
   * - `testPytorch.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testPytorch.py>`_
     - Provides an example of how to use a Pytorch Neural net Model as the classifier. (testRaw.py also has a Pytorch example with a C-NN).
   * - `testRaw.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testRaw.py>`_
     - This example shows how raw time series across multiple channels can be used as an input by utilising to use a custom feature extractor class, then initialising PyBCI with a custom C-NN Pytorch model. The raw data from the data receiver thread comes in the form [samples, channels] but for the standard scaler we want the shape `[channels, samples]` so we transpose the the data accordingly. Multiple channels are also dropped (with the PsuedoLSLSreamGenerator in mind) to save computational complexity as raw time series over large windows can give a lot of parameters for the neural net to train.
   * - `testSimple.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSimple.py>`_
     - Provides the simplest setup, where no specific streams or epoch settings are given, all default to sklearn SVM classifier and `GlobalEpochSettings() <https://github.com/LMBooth/pybci/blob/main/pybci/Configuration/EpochSettings.py>`_.
   * - `testSklearn.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSklearn.py>`_
     - Similar to testSimple.py, but shows MLP as custom sklearn classifier. Also illustrates examples of how to set individual marker time windows and configure datastream channels, global epoch-window settings, and machine learning settings before connecting to BCI and switching between training and test modes.
   * - `testTensorflow.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflow.py>`_
     - Similar to testSimple.py, but allows a custom TensorFlow model to be used. Establishes a connection to BCI, starts training on received epochs, checks the classifier's accuracy, and then switches to test mode to predict the current marker.
