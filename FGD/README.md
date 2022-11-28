## Fréchet Gesture Distance (FGD)

Scripts to calculate FGD for the GENEA Gesture Generation Challenge 2022 submissions.
We follow the FGD implementation in [Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity (ACM TOG, 2020)](https://arxiv.org/abs/2009.02119).

### Environment
* Ubuntu 18.04, Python 3.6, Pytorch 1.8.0

### Run
1. Train an autoencoder. You can set `n_frames` in `train_AE.py` to change the number of frames in a sample. 
   ```bash
   $ python FGD/train_AE.py
   ```
2. Calculate FGD.
   ```bash
   $ python FGD/evaluate_FGD.py
   ```

### Results
The FGD values for three different `n_frames=30,60,90`. We also report FGD on raw data space similar to the one used in [No Gestures Left Behind: Learning Relationships between Spoken Language and Freeform Gestures (EMNLP Findings 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.170.pdf). Lower FGD is better.

Autoencoder checkpoints are available [here (to be uploaded)](). 

**Results for the full-body tier**
```text
----- Experiment (motion chunk length: 30) -----
FGDs on feature space and raw data space
FBT:   28.645, 8415.563
FNA:   -0.000,   -0.012
FSA:    3.175, 1345.204
FSB:   16.266, 9925.688
FSC:   16.366, 3647.519
FSD:   43.434, 9871.901
FSF:    7.490, 1718.182
FSG:   10.057, 2410.368
FSH:    4.024,  854.327
FSI:    4.865, 2041.908

----- Experiment (motion chunk length: 60) -----
FGDs on feature space and raw data space
FBT:   46.267, 17105.831
FNA:    0.000,   -0.081
FSA:    5.155, 2946.741
FSB:   29.422, 20356.742
FSC:   24.778, 7506.545
FSD:   39.925, 19943.333
FSF:   13.919, 3682.961
FSG:    9.804, 5221.833
FSH:    3.258, 1844.664
FSI:   13.745, 4183.702

----- Experiment (motion chunk length: 90) -----
FGDs on feature space and raw data space
FBT:   19.866, 25725.445
FNA:    0.000,   -0.200
FSA:    4.410, 4810.428
FSB:   14.919, 31158.921
FSC:   17.307, 11494.462
FSD:   19.973, 30142.942
FSF:    8.901, 5848.064
FSG:    9.643, 8272.341
FSH:    3.085, 2958.145
FSI:    4.807, 6450.567
```

**Results for the upper-body tier**
```
----- Experiment (motion chunk length: 30) -----
FGDs on feature space and raw data space
UBA:   16.952, 9477.682
UBT:   18.191, 7497.609
UNA:    0.000,   -0.010
USJ:    4.834, 2307.812
USK:   15.472, 2887.649
USL:   28.435, 6715.159
USM:    2.171, 1008.365
USN:  194.180, 33632.683
USO:   16.432, 3117.656
USP:   17.955, 8462.732
USQ:    2.841, 1089.384

----- Experiment (motion chunk length: 60) -----
FGDs on feature space and raw data space
UBA:   11.494, 19120.322
UBT:   16.079, 15269.160
UNA:    0.000,   -0.072
USJ:    4.618, 4711.649
USK:   14.929, 5920.969
USL:   26.198, 13519.083
USM:    1.658, 2203.772
USN:  170.753, 67444.063
USO:   16.062, 6424.674
USP:   16.708, 17377.555
USQ:    2.536, 2400.012

----- Experiment (motion chunk length: 90) -----
FGDs on feature space and raw data space
UBA:   10.639, 28869.044
UBT:   12.363, 22969.446
UNA:    0.000,   -0.181
USJ:    4.463, 7222.822
USK:   14.196, 9051.976
USL:   22.885, 20369.668
USM:    1.127, 3558.372
USN:   35.853, 101443.302
USO:   15.005, 9844.623
USP:   11.006, 26615.193
USQ:    2.502, 3935.777
```