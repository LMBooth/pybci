.. _examples:

Examples
========

The following examples can all be found on the `PyBCI GitHub repository <https://github.com/LMBooth/pybci/tree/main/pybci/Examples>`_.

.. note:: 
   The examples have shields describing whether they work with PyBCI's pseudoDevice class and what additional external hardware is required. Pseudo Device description found here: :ref:`what-pseudo-device`
   
   If using your own LSL-capable hardware  and marker stream set :py:data:`createPseudoDevice=False` or optionally pass True or False as an arguement to each script.

PyBCI requires an LSL marker stream for defining when time series data should be attributed to an action/marker/epoch and an LSL data stream to create time-series data. 

If the user has no available LSL hardware to hand they can set :py:data:`createPseudoDevice=True` when instantiating the PyBCI object to enable a pseudo LSL data stream to generate time-series data and LSL marker stream for epoching the data. More information on PyBCI's Pseudo Device class can be found here: :ref:`what-pseudo-device`. 

The `example scripts <https://pybci.readthedocs.io/en/latest/BackgroundInformation/Examples.html>`_ illustrate various applied ML libraries (SKLearn, Tensorflow, PyTorch) or provide examples of how to integrate LSL hardware.

The code snippet can be used below to run a simple classification task using the Pseudo Device, alternatively call pybci in the command line to get a list of CLI commands and tests:


ArduinoHandGrasp
----------------
.. image:: https://img.shields.io/badge/Pseudo_Device-Not_Available-red
   :alt: pseudo device not available shield
.. image:: https://img.shields.io/badge/Arduino-Required-yellow
   :alt: arduino required shield
.. image:: https://img.shields.io/badge/Myoware_Muscle_Sensor-Required-yellow
   :alt: Myoware required shield
   
- **GitHub Link**: `ArduinoHandGrasp/ <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/ArduinoHandGrasp>`_
- **Description**: This folder contains an LSL marker creator in `MarkerMaker.py`, which uses PyQt5 as an on-screen text stimulus. It also includes `ServoControl.ino`, designed for an Arduino Uno to control 5 servo motors. A `Myoware Muscle Sensor` is attached to analog pin A0. The `ArduinoToLSL.py` script sends and receives serial data, while `testArduinoHand.py` classifies the data.

PupilLabsRightLeftEyeClose
--------------------------
.. image:: https://img.shields.io/badge/Pseudo_Device-Not_Available-red
   :alt: pseudo device not available shield
.. image:: https://img.shields.io/badge/Pupil_Labs_Hardware-Required-yellow
   :alt: pupil required shield

- **GitHub Link**: `PupilLabsRightLeftEyeClose/ <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/>`_
- **Description**: This folder contains a basic Pupil Labs example with a custom extractor class. `RightLeftMarkers.py` uses Tkinter to generate visual stimuli. `bciGazeExample.py` shows how a custom feature extractor class can be used.

MultimodalPupilLabsEEG
-----------------------
.. image:: https://img.shields.io/badge/Pseudo_Device-Not_Available-red
   :alt: pseudo device not available shield
.. image:: https://img.shields.io/badge/Pupil_Labs_Hardware-Required-yellow
   :alt: pupil required shield
.. image:: https://img.shields.io/badge/ioBio_EEG_Device-Required-yellow
   :alt: iobio EEG device required shield

- **GitHub Link**: `MultimodalPupilLabsEEG/ <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/MultimodalPupilLabsEEG>`_
- **Description**: An advanced example illustrating the use of two devices: Pupil Labs and Hull University ioBio EEG device. Includes a YouTube video demonstrating the multimodal example.

testEpochTimingsConfig
-----------------------
.. image:: https://img.shields.io/badge/Pseudo_Device-Available-blue
   :alt: pseudo device not available shield
- **GitHub Link**: `testEpochTimingsConfig <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testEpochTimingsConfig.py>`_
- **Description**: A simple example showing custom global epoch settings.

testPytorch
-----------
.. image:: https://img.shields.io/badge/Pseudo_Device-Available-blue
   :alt: pseudo device not available shield
- **GitHub Link**: `testPytorch <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testPyTorch.py>`_
- **Description**: Provides an example of using a PyTorch Neural Net Model as the classifier.

testRaw
-------
.. image:: https://img.shields.io/badge/Pseudo_Device-Available-blue
   :alt: pseudo device not available shield
- **GitHub Link**: `testRaw <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testRaw.py>`_
- **Description**: Demonstrates how raw time series data can be used as an input by utilizing a custom feature extractor class.

testSimple
----------
.. image:: https://img.shields.io/badge/Pseudo_Device-Available-blue
   :alt: pseudo device not available shield
- **GitHub Link**: `testSimple <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSimple.py>`_
- **Description**: Provides the simplest setup with default settings.

testSklearn
-----------
.. image:: https://img.shields.io/badge/Pseudo_Device-Available-blue
   :alt: pseudo device not available shield
- **GitHub Link**: `testSklearn <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testSklearn.py>`_
- **Description**: Similar to `testSimple`, but uses an MLP as a custom classifier.

testTensorflow
--------------
.. image:: https://img.shields.io/badge/Pseudo_Device-Available-blue
   :alt: pseudo device not available shield
- **GitHub Link**: `testTensorflow <https://github.com/LMBooth/pybci/blob/main/pybci/Examples/testTensorflow.py>`_
- **Description**: Similar to `testSimple`, but allows for a custom TensorFlow model to be used.
