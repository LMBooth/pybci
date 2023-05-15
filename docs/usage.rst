Usage
=====

.. _installation:

Installation
------------

To use PyBCI, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pybci



For example:

>>> from pybci import PyBCI
>>> bci = PyBCI()
>>> if bci.lslScanner.CheckAvailableLSL():
>>>     bci.StartTraining()

