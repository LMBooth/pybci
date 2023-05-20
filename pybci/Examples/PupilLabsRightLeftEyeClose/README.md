# Pupil Labs Right Left Eye Close Example

The Pupil Labs Right Left Eye Close Example in (bciGazeExample.py)[https://github.com/LMBooth/pybci/blob/main/pybci/Examples/PupilLabsRightLeftEyeClose/bciGazeExample.py] illustrates how a 'simple' custom pupil-labs feature extractor class can be passed for the gaze data, where the mean pupil diameter is taken for each eye and both eyes and used as feature data, where nans for no confidence are set to a value of 0.

The RightLeftMarkers.py script generates LSLMarkers and an onscreen stimulus in pythons built-in tkinter to inform the user when to shut what eye for training the bci.

It's advised to have a play with the ustom decoding class PupilGazeDecode() and see if other filtering/feature extraction methods can be used to improve the classifier.
