Welcome to the PyBCI documentation!
===================================

**PyBCI** is a Python interface to create a BCI (Brain Computer Interface) with the `Lab Streaming Layer <https://github.com/sccn/labstreaminglayer.>`_, `scikit-learn <https://scikit-learn.org/stable/#>`_ and `TensorFlow <https://www.tensorflow.org/install>`_ packages. Also leveraging packages like `Antropy <https://github.com/raphaelvallat/antropy>`_, `SciPy <https://scipy.org/>`_ and `NumPy <https://numpy.org/>`_ for time and/or frequency based feature exraction.

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
   BackgroundInformation/Classification

.. toctree::
   :maxdepth: 1
   :caption: API

   api/PyBCI
   api/LSLScanner
