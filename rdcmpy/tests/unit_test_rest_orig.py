"""Test the original rDCM model with resting-state data"""

from pathlib import Path

from scipy.io import loadmat

from rdcmpy import RegressionDCM

file_dir = Path(__file__).resolve().parent

data = loadmat(f'{file_dir}/unittest_rsorig.mat')
t_rep = 1.25

rdcm = RegressionDCM(data=data['data'], t_rep=t_rep, method='original', debug=True)
rdcm.estimate()
params = rdcm.get_params()

# check estimated effective connectivity
diff = data['mu_conn'] - params['mu_connectivity']
print(f'Difference in connectivity: total = {diff.sum()}, mean = {diff.mean()}')
