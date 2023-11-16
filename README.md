# LER_prediction_intervals
Prediction Intervals for Line Edge Roughness (LER) Estimation. This code has been developed by Inimfon Akpabio and 
Dr. Serap Savari. All rights reserved.

The code in this repo is organized into 3 folders each corresponding to research work/publications on constructing 
prediction intervals for Line Edge Roughness (LER) Estimation.
- spie_jm3: https://doi.org/10.1117/1.JMM.20.4.041206
- asmc: https://doi.org/10.1109/ASMC54647.2022.9792521
- ieee_tsm: https://doi.org/10.1109/TSM.2023.3270230

Each of the techniques for interval construction presented here build upon the prior LER estimation work by 
Dr. Narendra Chaudhary in https://github.com/narendrachaudhary51/LER_machine_learning

## Image Datasets
The single line original image dataset can be downloaded from this link - 
https://drive.google.com/a/tamu.edu/file/d/13_u6IpFfprnCmfy82vYsGRov6YTbdl56/view?usp=sharing

The single line noisy image dataset can be downloaded from this link - 
https://drive.google.com/a/tamu.edu/file/d/1DTbKKd9GSLHMbx_3IxiBzs7LWgT7IusZ/view?usp=sharing

The linescan dataset can be downloaded from this link - 
https://drive.google.com/a/tamu.edu/file/d/11LcLFm-cmUwHwLG1HC9Ie0l2vckrsEuc/view?usp=sharing

## Results
Each publication folder should have a `models_coverage_statistics.ipynb` file which is an interactive python notebook. 
Most of the results from the corresponding publication can be reproduced by running the code in these notebooks (Note 
that the filepaths will have to be modified for this to work). Furthermore, the experimental test data needed to run 
these notebooks have been pre-computed and stored in serialized pandas dataframe files (located in `/models` 
sub-directory of each publication directory).