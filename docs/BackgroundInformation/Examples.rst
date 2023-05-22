.. _examples:
Examples
############

.. list-table:: PyBCI Examples
   :widths: 25 75
   :header-rows: 1

   * - Example File
     - Description
   * - `PsuedoLSLSreamGenerator/mainSend.py <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/PsuedoLSLSreamGenerator/mainSend.py>`_
     - Generates multiple channels on a given stream type at a given sample rate. A baseline signal is generated on an LSL stream outlet and a PyQt button can be pressed to signify this signal on a separate LSL marker stream. The signal can be altered by 5 distinct markers for a configurable amount of time, allowing the user to play with various signal patterns for clasification. NOTE: Requires `PyQt5` and `pyqtgraph` installs for data viewer.
   * - `PupilLabs/bciGazeExample.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabs/bciGazeExample.py>`_
     - Illustrates how a 'simple' custom pupil-labs feature extractor class can be passed for the gaze data, where the mean pupil diameter is taken for each eye and both eyes and used as feature data, where nans for no confidence are set to a value of 0.
   * - `PupilLabs/RightLeftMarkers.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabs/RightLeftMarkers.py>`_
     - Uses tkinter to generate visual on-screen stimuli for only right, left or both eyes open, sends same onscreen stimuli as LSL markers, ideal for testing pupil-labs eyes classifier test.
   * - `testEpochTimingsConfig.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testEpochTimingsConfig.py>`_
     - Simple example showing custom global epoch settings  changed on initialisation.
   * - `testSimple.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSimple.py>`_
     - Provides the simplest setup, where no specific streams or epoch settings are given, all default to sklearn SVM classifier and GeneralEpochSettings.
   * - `testSklearn.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSklearn.py>`_
     - Similar to testSimple.py, but allows a custom sklearn classifier to be used. It sets individual time windows and configures data stream channels, epoch window settings, and machine learning settings before connecting to BCI and switching between training and test modes.
   * - `testTensorflow.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflow.py>`_
     - Similar to testSimple.py, but allows a custom TensorFlow model to be used. It establishes a connection to BCI, starts training on received epochs, checks the classifier's accuracy, and then switches to test mode to predict the current marker.
   * - `testTensorflowRaw.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflowRaw.py>`_
     - Similar to testTensorflow.py but allows for a custom tensorflow model to be used with raw data. This example shows how to create a custom tensorflow model, use a custom feature extractor class, and then initialize PyBCI with the custom model. The raw data has he shape `[ch, samps]` where the ch is the number of channels and samps is the number of samples in the given time window.

