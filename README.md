# Todd-Nitz-ProjectSp2023
Matthew Todd's Spring 2023 Project under Advisor (Alex Nitz).

`simulator.py` takes two parameters (A = Amplitude, B = frequency), and plots n points of a sine wave
with bounds (default = 0,2pi), with or without noise.

`train1p.py` uses simulation file for different frequencies and compares to initial sine wave ("observed data"), and outputs
frequency parameter that best matches the observed curve. This file has a fixed amplitude, that is "known" to the model.

`train2p.py` uses simulation file for different frequencies and compares to initial sine wave ("observed data"), and outputs
amplitude and frequency parameters that best matches the observed curve. In this file, both paramters (amplitude and frequency) are unkownn
to the model.
