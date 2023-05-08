Usage
=====

.. _installation:

Installation
------------

To use PyBCI, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pybci



For example:

>>> from pylslbci import PyLSLBCI
>>> bci = PyLSLBCI()
>>> if bci.lslScanner.CheckAvailableLSL():
>>>     bci.StartTraining()

