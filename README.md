# EEG-BCI

The aim of this project is to develop a solution that lowers the rates of injuries by death.

The main technology is supported by EEG BCI's that use FES to simulate muscles when individuals lose fine control over them.

Currently looking for any BCIs that cover the cz, c3, and c4 areas

Where I am at:-
AlphaWaveDetector initializes to 10Hz as those are alpha waves from physics

In the convolved return type from AlphaWaveDetector, if the value is high we know that we are detecting alpha waves.

Currently in the midst of figuring out what waves to detect since Muse does not detect PePs. Once this parametrization is figured out I can set the initial convolved filter to match that

*Pooling needs to be implemented after making a FFT function*