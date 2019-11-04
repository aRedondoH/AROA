# AROA

This repository contains the code and the data used for testing the experiments of AROA approach. It contains three diffent directories:
  
* Analysis: this directory contains the jupyter notebook files used to create the plots
* Data: this directory contains the data necesary to reproduce the results obtained
* src: this directory contains the code neccesary to run the experiment through the cluster

### Prerequisites
Following packages need to be installed

```
numpy
pandas
sklearn
joblib
multiprocessing
scipy
```
* Python 3.x.

### Running
Configurations in expLauncher.py:
* varBeta: variance of the beta distribution
* K: number of simulations
* nOri: number of originals to generate
* nAtt: number of attacks to generate
* feaToCheck: number of features (from the originals generated) to change from 1 to 0
* feaToAttack: number of features to attack per binary to change the features from 0 to 1
* nTriesFindAttack: number of tries to find the attack from the binary obtained from test set
* numCores: number of cores to be used in the cluster

- expCombinations() function performs different parameters combinations
- expSpecific() function perform n experiments with a specific configuration
```
python expLauncher.py
```

The program write the features extracted from the binaries into a .csv file in Output folder
