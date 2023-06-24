Welcome to the PyBCI documentation!
===================================

**PyBCI**  is a Python package to create a Brain Computer Interface (BCI) with data synchronisation and pipelining handled by the `Lab Streaming Layer <https://github.com/sccn/labstreaminglayer>`_, machine learning with  `Pytorch <https://pytorch.org/>`_, `scikit-learn <https://scikit-learn.org/stable/#>`_ and `TensorFlow <https://www.tensorflow.org/install>`_, leveraging packages like `Antropy <https://github.com/raphaelvallat/antropy>`_, `SciPy <https://scipy.org/>`_ and `NumPy <https://numpy.org/>`_ for generic time and/or frequency based feature extraction or optionally have the users own custom feature extraction class used.

The goal of PyBCI is to enable quick iteration when creating pipelines for testing human machine and brain computer interfaces, namely testing applied data processing and feature extraction techniques on custom machine learning models. Training the BCI requires LSL enabled devices and an LSL marker stream for training stimuli. (The `examples folder <https://github.com/LMBooth/pybci/tree/main/pybci/Examples>`_ found on the github has a `pseudo LSL data generator and marker creator <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/PsuedoLSLStreamGenerator>`_ in the `mainSend.py <https://github.com/LMBooth/pybci/tree/main/pybci/Examples/PsuedoLSLStreamGenerator/mainSend.py>`_ file so the examples can run without the need of LSL capable hardware.)



Check out the :doc:`BackgroundInformation/Introduction` section for further information, specifically for :ref:`installation` of the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: User's Guide

   BackgroundInformation/Introduction
   BackgroundInformation/Theory_Operation
   BackgroundInformation/Epoch_Timing
   BackgroundInformation/Feature_Selection
   BackgroundInformation/Examples

.. toctree::
   :maxdepth: 1
   :caption: API

   api/PyBCI
   api/LSLScanner
   api/Configurations
