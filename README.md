# rDCM toolbox in Python
Python implementation of the 
[regression Dynamic Causal Modelling (rDCM) toolbox](https://github.com/translationalneuromodeling/tapas/tree/master/rDCM) 
(v6.0.0).

## Installation
```python
python3 -m pip install git+https://github.com/jadecci/rDCM_py.git
```

## Usage
```python
from rdcmpy import RegressionDCM

rdcm = RegressionDCM(data, TR, drive_input=task_regressors, prior_a=SC)
rdcm.estimate()
params = rdcm.get_params()

A = params['mu_connectivity']
C = params['mu_driving_input']
```

### Functions translated
1. Original ridge rDCM model
2. Works for both task and resting-state fMRI data
3. Works for real data

### Functions yet missing
1. Sparse rDCM model
2. Option to use synthetic/simulated data
3. Option to create covaraince matrix
4. Option to predict signals (in time domain) and evaluate the prediction

# References

1. Frässle, S., Lomakina, E.I., Razi, A., Friston, K.J., Buhmann, J.M., Stephan, K.E., 2017. 
Regression DCM for fMRI. *NeuroImage* 155, 406–421. 
doi: [10.1016/j.neuroimage.2017.02.090](https://doi.org/10.1016/j.neuroimage.2017.02.090)
2. Frässle, S., Lomakina, E.I., Kasper, L., Manjaly Z.M., Leff, A., Pruessmann, K.P., 
Buhmann, J.M., Stephan, K.E., 2018. A generative model of whole-brain effective connectivity. 
*NeuroImage* 179, 505-529. 
doi: [10.1016/j.neuroimage.2018.05.058](https://doi.org/10.1016/j.neuroimage.2018.05.058)

### rDCM for resting-state fMRI

3. Frässle, S., Harrison, S.J., Heinzle, J., Clementz, B.A., Tamminga, C.A., Sweeney, J.A., 
Gershon, E.S., Keshavan, M.S., Pearlson, G.D., Powers, A., Stephan, K.E., 2021. 
Regression dynamic causal modeling for resting-state fMRI. *Human Brain Mapping* 42, 2159-2180. 
doi: [10.1002/hbm.25357](https://doi.org/10.1002/hbm.25357)

### SPM DCM fMRI prior

4. Marreiros AC, Kiebel SJ, Friston KJ. 2008. Dynamic causal modelling for fMRI: a two-state model.
*Neuroimage* 39, 269-78.
5. Stephan KE, Kasper L, Harrison LM, Daunizeau J, den Ouden HE, Breakspear M, Friston KJ. 2008. 
Nonlinear dynamic causal models for fMRI. *Neuroimage* 42, 649-662.
