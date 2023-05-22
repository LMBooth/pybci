.. _examples:
Examples
############

.. list-table:: PyBCI Examples
   :widths: 25 75
   :header-rows: 1

   * - Example File
     - Description
   * - `PsuedoLSLSreamGenerator/mainSend.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PsuedoLSLSreamGenerator/mainSend.py>`_
     - Generates multiple channels on a given stream type at a given sample rate. A baseline signal is generated on an LSL stream outlet and a PyQt button can be pressed to signify this baseline signal on a separate LSL marker stream.
   * - `PupilLabs/bciGazeExample.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabs/bciGazeExample.py>`_
     - Illustrates how a 'simple' custom pupil-labs feature extractor class can be passed for the gaze data, where the mean pupil diameter is taken for each eye and both eyes and used as feature data, where nans for no confidence are set to a value of 0.
   * - `testSimple.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSimple.py>`_
     - Provides the simplest setup, where no specific streams or epoch settings are given, all default to sklearn SVM classifier and GeneralEpochSettings.
   * - `testSklearn.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSklearn.py>`_
     - Similar to testSimple.py, but allows a custom sklearn classifier to be used. It sets individual time windows and configures data stream channels, epoch window settings, and machine learning settings before connecting to BCI and switching between training and test modes.
   * - `testTensorflow.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflow.py>`_
     - Similar to testSimple.py, but allows a custom TensorFlow model to be used. It establishes a connection to BCI, starts training on received epochs, checks the classifier's accuracy, and then switches to test mode to predict the current marker.
   * - `testTensorflowRaw.py <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflowRaw.py>`_
     - Similar to testTensorflow.py but allows for a custom tensorflow model to be used with raw data. This example shows how to create a custom tensorflow model, use a custom feature extractor class, and then initialize PyBCI with the custom model.
