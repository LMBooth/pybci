# PyBCI Examples

This folder holds multiple scripts illustrating the functions and configurations available within the PyBCI package.

## Example descriptions
### The following files will all work with the mainSend.py file found in the PsuedoLSLSreamGenerator folder. 
- testSimple.py - Gives bare bones, simplest setup, where no specified streams or epoch settins are given, all default to sklearn SVM classifier and GeneralEpochSettings
- testSklearn.py - Similar to testSimple.py, but allows custom sklearn clf to be used.
- testTensorflow.py - Similar to testSimple.py, but allows custom tensorflow model to be used.
- 

## Specific Examples
- A simple eample for Pupil labs has been created in the PupilLabsRightLeftEyeClose showing how a custom class for interpreting pupil labs lsl relay data.
