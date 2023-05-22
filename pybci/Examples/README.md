# PyBCI Examples

This folder holds multiple scripts illustrating the functions and configurations available within the PyBCI package.

| File/Subfolder | Description |
| --- | --- |
| PsuedoLSLSreamGenerator/mainSend.py | Generates multiple channels on a given stream type at a given sample rate. A baseline signal is generated on an LSL stream outlet and a PyQt button can be pressed to signify this baseline signal on a separate LSL marker stream. The script requires PyQt5 for the button interface and PyQtGraph for data plotting. The signal for each marker type can be modified by changing the PsuedoEMGDataConfig properties in the init for each marker.|
| Pupil Labs Right Left Eye Close Example/bciGazeExample.py | Illustrates how a 'simple' custom pupil-labs feature extractor class can be passed for gaze data. The mean pupil diameter is taken for each eye and both eyes and used as feature data. It's advised to experiment with the custom decoding class PupilGazeDecode() in the script to see if other filtering/feature extraction methods can improve the classifier. |
| Pupil Labs Right Left Eye Close Example/RightLeftMarkers.py | Generates LSLMarkers and an onscreen stimulus in Python's built-in tkinter to inform the user when to shut which eye for training the BCI. |
| testSimple.py | Gives a bare-bones setup, where no specified streams or epoch settings are given. All default to the sklearn SVM classifier and GeneralEpochSettings. |
| testSklearn.py | Similar to testSimple.py, but allows a custom sklearn classifier to be used. |
| testTensorflow.py | Similar to testSimple.py, but allows a custom TensorFlow model to be used. |
| testMarkerStream.py | This script shows how to set up and run the PyBCI package. It illustrates how to configure data stream channels, epoch window settings, and how to connect, train, and test with the PyBCI class. It also provides examples of how to use other functions in the PyBCI package. |
| testSimpleAPI.py | This script provides a simple example of how to use the PyBCI package. It demonstrates how to connect to the LSL marker and data stream, start training on received epochs, check the number of received epochs, get the accuracy of the classifier, and get the current marker estimation. It also shows how to switch from train mode to test mode based on the number of received markers. |
