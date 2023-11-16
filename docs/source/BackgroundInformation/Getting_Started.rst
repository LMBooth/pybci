Getting Started
###############



Python Package Dependencies Version Minimums
============================================
PyBCI is tested on Python versions 3.9, 3.10 and 3.11 (`defined via appveyor.yml <https://github.com/LMBooth/pybci/blob/main/appveyor.yml>`__)

The following package versions define the minimum supported by PyBCI, also defined in  :py:data:`setup.py`:

.. code-block:: console

    pylsl>=1.16.1,
    scipy>=1.11.1,
    numpy>=1.24.3,
    antropy>=0.1.6,
    tensorflow>=2.13.0,
    scikit-learn>=1.3.0,
    torch>=2.0.1
    
Earlier packages may work but are not guaranteed to be supported.

Prerequisite for Non-Windows Users
==================================
If you are not using windows then there is a prerequisite stipulated on the `pylsl repository <https://github.com/labstreaminglayer/pylsl#prerequisites>`_ to obtain a liblsl shared library. See the `liblsl repo documentation <https://github.com/sccn/liblsl>`_ for more information. Once the liblsl library has been downloaded pip install pybci-package should work.

.. _installation:
Installation
============

For stable releases use:  :py:data:`pip install pybci-package`

For development versions use:  :py:data:`pip install git+https://github.com/LMBooth/pybci.git` or 

.. code-block:: console

   git clone https://github.com/LMBooth/pybci.git
   cd pybci
   pip install -e .


Optional: Virtual Environment
-----------------------------
Or optionally, install and run in a virtual environment:

Windows:

.. code-block:: console

   python -m venv my_env
   .\my_env\Scripts\Activate
   pip install pybci-package  # For stable releases
   # OR
   pip install git+https://github.com/LMBooth/pybci.git  # For development version

Linux/MaxOS:

.. code-block:: console

   python3 -m venv my_env
   source my_env/bin/activate
   pip install pybci-package  # For stable releases
   # OR
   pip install git+https://github.com/LMBooth/pybci.git  # For development version

There has been issues raised with Linux successfully running all pytests and examples, there is a dockerfile included in the root repository outlining what should be a successful build of ubuntu 22:04.

Dockerfile
======================
There is an Ubuntu 22.04 setup found in the `Dockerfile <https://github.com/LMBooth/pybci/blob/main/Dockerfile>`__ in the root of the directory which can be used in conjunction with `docker <https://docs.docker.com/desktop/>`__ .

Once docker is installed call the following in the root directory:

.. code-block:: console

    sudo docker build -t pybci .
    sudo docker run -it -p 4000:8080 pybci

Then either run the :console:data:`pybci` CLI command or run :console:data:`pytest Tests`.

Download the Dockerfile and run 

Running Pytest Locally
======================
After installing pybci and downloading and extracting the pybci git repository, navigate to the extracted location and run  :py:data:`pip install requirements-devel.txt` to install pytest, then call  :py:data:`pytest -vv -s Tests\` to run all the automated tests and ensure all 10 tests pass (should take approximately 15 mins to complete), this will ensure pybci functionality is as desired.

.. _simpleimplementation:

Simple Implementation
=====================
PyBCI requires an LSL marker stream for defining when time series data should be attributed to an action/marker/epoch and an LSL data stream to create time-series data. 

If the user has no available LSL hardware to hand they can set  :py:data:`createPseudoDevice=True` when instantiating the PyBCI object to enable a pseudo LSL data stream to generate time-series data and LSL marker stream for epoching the data. More information on PyBCI's Pseudo Device class can be found here: :ref:`what-pseudo-device`. 

The `example scripts <https://pybci.readthedocs.io/en/latest/BackgroundInformation/Examples.html>`_ illustrate various applied ML libraries (SKLearn, Tensorflow, PyTorch) or provide examples of how to integrate LSL hardware.

The code snippet can be used below to run a simple classification task using the Pseudo Device, alternatively call pybci in the command line to get a list of CLI commands and tests:

.. code-block:: python

   from pybci import PyBCI
   import time 
   
   if __name__ == '__main__': # Note: this line is needed when calling pseudoDevice as by default runs in a multiprocessed operation
       bci = PyBCI(minimumEpochsRequired = 5, createPseudoDevice=True)
       while not bci.connected: # check to see if lsl marker and datastream are available
           bci.Connect()
           time.sleep(1)
       bci.TrainMode() # now both marker and datastreams available start training on received epochs
       accuracy = 0
       try:
           while(True):
               currentMarkers = bci.ReceivedMarkerCount() # check to see how many received epochs, if markers sent to close together will be ignored till done processing
               time.sleep(0.5) # wait for marker updates
               print("Markers received: " + str(currentMarkers) +" Accuracy: " + str(round(accuracy,2)), end="         \r")
               if len(currentMarkers) > 1:  # check there is more then one marker type received
                   if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired:
                       classInfo = bci.CurrentClassifierInfo() # hangs if called too early
                       accuracy = classInfo["accuracy"]
                   if min([currentMarkers[key][1] for key in currentMarkers]) > bci.minimumEpochsRequired+10:  
                       bci.TestMode()
                       break
           while True:
               markerGuess = bci.CurrentClassifierMarkerGuess() # when in test mode only y_pred returned
               guess = [key for key, value in currentMarkers.items() if value[0] == markerGuess]
               print("Current marker estimation: " + str(guess), end="           \r")
               time.sleep(0.2)
       except KeyboardInterrupt: # allow user to break while loop
           print("\nLoop interrupted by user.")

