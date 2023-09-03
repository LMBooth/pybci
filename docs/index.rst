Welcome to the PyBCI documentation!
===================================

**PyBCI**  is a Python package to create a Brain Computer Interface (BCI) with data synchronisation and pipelining handled by the `Lab Streaming Layer <https://github.com/sccn/labstreaminglayer>`_, machine learning with  `Pytorch <https://pytorch.org/>`_, `scikit-learn <https://scikit-learn.org/stable/#>`_ or `TensorFlow <https://www.tensorflow.org/install>`_, leveraging packages like `Antropy <https://github.com/raphaelvallat/antropy>`_, `SciPy <https://scipy.org/>`_ and `NumPy <https://numpy.org/>`_ for generic time and/or frequency based feature extraction or optionally have the users own custom feature extraction class used.

The goal of PyBCI is to enable quick iteration when creating pipelines for testing human machine and brain computer interfaces, namely testing applied data processing and feature extraction techniques on custom machine learning models. Training the BCI requires LSL enabled devices and an LSL marker stream for training stimuli. All the `examples <https://github.com/LMBooth/pybci/tree/main/pybci/Examples>`__ found on the github not in a dedicated folder have a pseudo LSL data generator enabled by default, by setting `createPseudoDevice=True` so the examples can run without the need of LSL capable hardware.

`Github repo here! <https://github.com/LMBooth/pybci/>`_



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
   BackgroundInformation/Pseudo_Device
   BackgroundInformation/Examples

.. toctree::
   :maxdepth: 1
   :caption: API

   api/PyBCI
   api/LSLScanner
   api/PseudoDeviceController
   api/PseudoDevice
   api/Configurations
