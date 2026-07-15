# Hydro-Numerics-1D
Tools for solving the 1D hydrodynamics equations by finite differences

# Overview 

This repository is a collection of tools for numerically solving the 1D hydrodynamic 
equations. It is developed in service of the following publications: 

Wolf. E. A. and Zwierlein, M. arXiv:2606.06659 (submitted to PRL)
https://doi.org/10.48550/arXiv.2606.06659


The code is in large part detail-agnostic, but some decisions have been made with the specific case 
of our experimental system - a unitary Fermi gas in a box trap - in mind: 

## Normalizations

Simulations are conducted in dimensionless units. For many problems, these may be taken to correspond 
to units in which the initial density and speed of sound of the gas are normalized to 1. 


## Polytropic Equation of State 

The unitary Fermi gas at constant entropy features a polytropic equation of state, P = K rho^gamma, 
with gamma = 5/3. As such, there is substantial support for the special case of a polytropic EOS. 

## Isentropic Dynamics 

The evolution of a 1D system of Euler equations is both adiabatic and locally isentropic. The code will consider this a base case, with entropy dynamics an add-on. 

# Disclaimer

I (EW) am not a numerics expert. It is not my fault if this code doesn't do what you want. It is also not my fault if this code fails to do what you want in a way that is not immediately conspicuous. 

Some tools will be included to try to benchmark numerical stability, but I do not envision any automated warnings, smart parameter checking, or detailed analysis of discretization errors.


# Acknowledgements 

Inspiration for and insight into numerical techniques to use was drawn from the following sources: 

https://fab.cba.mit.edu/classes/864.17/text/pde_diff.pdf


