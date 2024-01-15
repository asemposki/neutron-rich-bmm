# Bayesian Model Mixing and the dense matter equation of state: an exploration of symmetric matter

## About 

A joint effort between members of the Bayesian Analysis of Nuclear Dynamics (BAND) collaboration and Bayesian Uncertainty Quantification (Errors in Your EFT) 
(BUQEYE) collaboration to perform principled uncertainty quantification of the dense matter equation of state (EOS) using the novel techniques 
in Bayesian Model Mixing (BMM). 

## Navigation

The following notebooks are currently included in this repo:

1) Pressure_Mixing_cs2.ipynb : this notebook runs the Bayesian model mixing of the results from ChEFT and pQCD. Includes the speed of sound at the end of the notebook.

2) derivatives-bands.ipynb : this is the old notebook from nuclear-matter-convergence, reworked so that the ChEFT results can be run with extrapolated truncation errors. The class file where the changes were made, derivatives.py, is not currently within the repo (yet).

3) pQCD_Gorda_gsum.ipynb : this notebook contains all current work for the pQCD EOS, run first with respect to chemical potential, and then with respect to density via the Kohn, Luttinger, and Ward (KLW) inversion. Speed of sound is calculated at the end of this notebook, and results are saved for use in Pressure_Mixing_cs2.ipynb.

## Contacts

Authors: Alexandra C. Semposki (Ohio U), Christian Drischler (Ohio U/FRIB), Richard J. Furnstahl (OSU), Jordan A. Melendez (OSU), and Daniel R. Phillips (Ohio U).
