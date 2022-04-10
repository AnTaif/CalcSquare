# CalcSquare
A script to calculate the area of an object of any shape by the area of a reference object (an object whose area we already know).

The first thing to do is to make sure that the reference object is to the left of all the others.
The mainCalcSquare function inputs an image (it is necessary that the objects to be calculated stand out from the background). 
The output is a list with the areas of the required objects (there may be several of them), and an image named "output-image.jpg" is created.

This script uses the libraries: numpy, matplotlib, scipy, imutils, opencv-python
--------------------------------------------------------------------------------
