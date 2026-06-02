### IMPORTS ###

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/..")
import argparse

from gp import GaussianProcess
import pandas as pd
import jax
import jax.numpy as jnp

N_CHAINS = 2
N_SAMPLES = 111
BURNIN = 10
THIN = 10

### JOB ARRAY INDEX ###
parser = argparse.ArgumentParser()
parser.add_argument("job_idx", type=int)
args = parser.parse_args()
job_idx = args.job_idx

param_grid = pd.read_csv("param_grid3.csv")
params = param_grid.iloc[job_idx]
ls_xy, ls_z, os_xyz, ls_t, os_t, ls_ap = (
    params["ls_xy"],
    params["ls_z"],
    params["os_xyz"],
    params["ls_t"],
    params["os_t"],
    params["ls_ap"],
)
ls_xyz = jnp.array([ls_xy, ls_xy, ls_z])

### DATA ###

X_train = jnp.load('X_train.npy')
y_train = jnp.load('y_train.npy')
obs_count = jnp.load('obs_count.npy')
obs_sse = jnp.load('obs_sse.npy') if os.path.exists('obs_sse.npy') else jnp.zeros_like(obs_count)

X_test = jnp.load('X_test.npy')
y_test = jnp.load('y_test.npy')

### GP SETUP ###

indoor_mask = X_train[:,2].astype(bool)
outdoor_mask = ~X_train[:,2].astype(bool)

indoor_mean = y_train[indoor_mask].mean()
outdoor_mean = y_train[outdoor_mask].mean()

def m(obs):
    return obs[2] * indoor_mean + (1 - obs[2]) * outdoor_mean

def K(obs1, obs2):
    d_xyz = ((obs1[:3] - obs2[:3])**2 / (2*ls_xyz**2)).sum()
    d_ap = jnp.where(obs1[4] == obs2[4], 0.0, 1.0) / (2*ls_ap**2)
    d_t = (obs1[3] - obs2[3])**2
    return os_xyz * jnp.exp(- (d_xyz + d_ap)) + \
        os_t * jnp.exp(- d_t / (2*ls_t**2))

### RUN GP ###

GP = GaussianProcess(m, K)
GP.fit(X_train, y_train, obs_count, obs_sse=obs_sse)
chain = GP.gibbs(chains=N_CHAINS, samples=N_SAMPLES)

cov_chains = chain[1][:,BURNIN::THIN,:]
new_means, new_vars = GP.predict(X_test, cov_chains)
mse = ((y_test - new_means.mean(axis=0))**2).mean()

results = f"{job_idx},{ls_xy},{ls_z},{os_xyz},{ls_t},{os_t},{ls_ap},{mse}"

with open(f"results/job_{job_idx}", "w", encoding="utf-8") as f:
    f.write(results)
