# Pupil Labs Right Left Eye Close Example

The Pupil Labs Right Left Eye Close Example in [bciGazeExample.py](https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/bciGazeExample.py) illustrates how a 'simple' custom pupil-labs feature extractor class can be passed for the gaze data, where the mean pupil diameter is taken for each eye and both eyes and used as feature data, where nans for no confidence are set to a value of 0.

The [RightLeftMarkers.py](https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/RightLeftMarkers.py) script generates LSLMarkers and an onscreen stimulus in pythons built-in tkinter to inform the user when to shut what eye for training the bci.

It's advised to have a play with the custom decoding class PupilGazeDecode() in [bciGazeExample.py](https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/bciGazeExample.py) and see if other filtering/feature extraction methods can be used to improve the classifier.

An example video using the multimodal example (pupil labs with Fp1 and Fp2 from EEG) can be found here:

[![PyBCI Multi-modal demo!](http://i3.ytimg.com/vi/SSmFU_Esayg/hqdefault.jpg)](https://www.youtube.com/watch?v=SSmFU_Esayg)
