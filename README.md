# From chiral EFT to perturbative QCD: a Bayesian model mixing approach to symmetric nuclear matter

## About 

A joint effort between members of the Bayesian Analysis of Nuclear Dynamics (BAND) collaboration and Bayesian Uncertainty Quantification (Errors in Your EFT) 
(BUQEYE) collaboration to perform principled uncertainty quantification of the dense matter equation of state (EOS) using the novel techniques 
in Bayesian model mixing (BMM). 

## Installing the repository and its dependencies

This repository is built to work off of the nuclear-matter-convergence repository of the BUQEYE collaboration, [found here](https://github.com/buqeye/nuclear-matter-convergence). As such, it has some dependencies that need to be built in the virtual environment before this repository can be run. Follow the steps below to complete this installation.

1. Create a new Conda environment: `conda create -n BUQEYE python==3.9.16`
2. Activate the environment: `conda activate BUQEYE`
3. Install jupyterlab and jupyter notebook: `conda install jupyterlab, jupyter notebook`
4. Clone this repository by copying the link in the code tab above: `git clone https://github.com/asemposki/EOS_BMM_SNM.git`
5. Within the above repository folder, git clone gsum: `git clone https://github.com/buqeye/gsum.git` and pip install this in gsum directory using `pip install .`
6. Also within the repository folder, git clone gptools: `git clone https://github.com/markchil/gptools.git` and pip install this in the gptools directory using `pip install .`
7. Still within this repository's main folder, git clone the nuclear-matter-convergence repo using `git clone https://github.com/buqeye/nuclear-matter-convergence.git` and `pip install .` within the nuclear-matter-convergence folder
8. Lastly, git clone the BAND package `Taweret` from `https://github.com/bandframework/Taweret.git` within the repository's main folder, as before
9. Now that this is all done, you can test the structure by going into the `notebooks` folder and saying `jupyter notebook` in terminal, and this should load up the notebooks to be run. You should be all set!

## Navigation

### Notebooks 
This folder contains all of the notebooks that generate our results in our paper. 

### data
All of our data is contained here.

### src
The source folder has all of the code written for this paper, including modified versions of open-source BUQEYE code from previous projects from the authors.

### Mathematica codes
Here you will find the Mathematica notebook that calculates the speeds of sound in Appendix B of the paper, and some of Appendix A as well.

## Contacts

Authors: Alexandra C. Semposki (Ohio U), Christian Drischler (Ohio U/FRIB), Richard J. Furnstahl (OSU), Jordan A. Melendez (OSU), and Daniel R. Phillips (Ohio U).
