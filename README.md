# CHOCOLATE
CHOCOLATE seizure diary simulator

Epilepsy Plus Data Science Lab
Daniel M Goldenholz 2022

This software will allow one to simulate a sythetic seizure diary, or a set of diaries.
The main code for simulation is in:
realSim.py

in there, the first function is called "simulator_base", and this is the recommended function to use in almost all cases. 
In reality it is a wrapper function for the simulate_dairy function with appropriate default values added in.

For the paper about this simulator, several additional files were used. None of these are needed to simulate your own synthetic patients, but if you wish to see how the paper's figures and examples were coded, see these.

prove_realsim.py - many helpful additional functions
trialSimulator.py -helpful functions specific to simulating clinical trials
for_the_paper.ipynb - jupytr notebook to build figures for the paper
forecastProof.ipynb - jupytr notebook to build deep learning example for the paper
