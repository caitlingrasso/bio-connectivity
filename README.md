# Revealing non-trivial information structures in aneural biological tissues via functional connectivity

Functional connectivity (FC) networks are constructed using bivariate mutual information of calcium traces from individual cells in *Xenopus laevis* organoids in both basal and perturbed states. Analysis of resulting FC networks reveal non-trivial, non-random information structures and suggest increased integration amongst cells in the organoids post-perturbation.  

This repository contains code for network inference and analysis (see below for video preprocessing). 

### Requirements:
1. Network inference requires the [Java Information Dynamics Toolkit (JIDT)](https://github.com/jlizier/jidt)
2. Network analysis requires the [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/)

### Video preprocessing:
1. [Image Registration](https://github.com/ELIFE-ASU/image-registration)
2. [Segmentation and signal extraction](https://github.com/caitlingrasso/calcium-signal-extraction)
