.. _examples:

Examples
========

The following examples can all be found on the `PyBCI GitHub repository <https://github.com/LMBooth/pybci/tree/main/pybci/Examples>`_.

.. note:: 
   All the examples shown that are not in a dedicated folder work with the `createPseudoDevice` variable set to `True` when instantiating `PBCI()`. If using your own LSL-capable hardware, you may need to adjust the scripts accordingly, namely set `createPseudoDevice=False`.

ArduinoHandGrasp
----------------
- **GitHub Link**: `ArduinoHandGrasp/ <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/ArduinoHandGrasp>`_
- **Description**: This folder contains an LSL marker creator in `MarkerMaker.py`, which uses PyQt5 as an on-screen text stimulus. It also includes `ServoControl.ino`, designed for an Arduino Uno to control 5 servo motors. A `Myoware Muscle Sensor` is attached to analog pin A0. The `ArduinoToLSL.py` script sends and receives serial data, while `testArduinoHand.py` classifies the data.

PupilLabsRightLeftEyeClose
--------------------------
- **GitHub Link**: `PupilLabsRightLeftEyeClose/ <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/>`_
- **Description**: This folder contains a basic Pupil Labs example with a custom extractor class. `RightLeftMarkers.py` uses Tkinter to generate visual stimuli. `bciGazeExample.py` shows how a custom feature extractor class can be used.

MultimodalPupilLabsEEG
-----------------------
- **GitHub Link**: `MultimodalPupilLabsEEG/ <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/MultimodalPupilLabsEEG>`_
- **Description**: An advanced example illustrating the use of two devices: Pupil Labs and Hull University ioBio EEG device. Includes a YouTube video demonstrating the multimodal example.

testEpochTimingsConfig
-----------------------
- **GitHub Link**: `testEpochTimingsConfig <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testEpochTimingsConfig.py>`_
- **Description**: A simple example showing custom global epoch settings.

testPytorch
-----------
- **GitHub Link**: `testPytorch <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testPytorch.py>`_
- **Description**: Provides an example of using a PyTorch Neural Net Model as the classifier.

testRaw
-------
- **GitHub Link**: `testRaw <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testRaw.py>`_
- **Description**: Demonstrates how raw time series data can be used as an input by utilizing a custom feature extractor class.

testSimple
----------
- **GitHub Link**: `testSimple <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSimple.py>`_
- **Description**: Provides the simplest setup with default settings.

testSklearn
-----------
- **GitHub Link**: `testSklearn <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSklearn.py>`_
- **Description**: Similar to `testSimple`, but uses an MLP as a custom classifier.

testTensorflow
--------------
- **GitHub Link**: `testTensorflow <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflow.py>`_
- **Description**: Similar to `testSimple`, but allows for a custom TensorFlow model to be used.
