# Hydro-Numerics-1D
Tools for solving the 1D hydrodynamics equations by finite differences

# Overview 

This repository is a collection of tools for numerically solving the 1D hydrodynamic 
equations. It is developed in service of the following publication: 

Wolf. E and Zwierlein, M. In prep. 

Efforts will be made to keep e.g. numerical steps detail-agnostic, but in general
the code will be highly tailored to the experimental system of relevance: a Riemann 
problem where the strongly-interacting Fermi gas is expanded into vacuum in a quasi-1D 
geometry. Where analytic results are coded up, they will be those of relevance to this system. 

Noteworthy design decisions in keeping with this include:

## Normalizations

For the Riemann problem of relevance, the initial mass density rho_0 and speed of sound c_0 are 
natural normalizations, and so will be used liberally to render units dimensionless. 

## Polytropic Equation of State 

The unitary Fermi gas at constant entropy features a polytropic equation of state, P = K rho^gamma, 
with gamma = 5/3. As such, the code will be designed with this case in mind - though where possible the option for general equations of state will be retained. 

## Isentropic Dynamics 

The evolution of a 1D system of Euler equations is both adiabatic and locally isentropic. The code will consider this a base case, with entropy dynamics an add-on. 

# Disclaimer

To begin: I am not a numerics expert. It is not my fault if this code doesn't do what you want. It is especially not my fault if this code fails to do what you want in a way that is not immediately conspicuous. 

For my own utility, some tools will be included to try to benchmark numerical stability, but I do not envision any automated warnings, smart parameter checking, or detailed analysis of discretization errors.

You have been warned. 


# Acknowledgements 

Inspiration for and insight into numerical techniques to use was drawn from the following sources: 

https://fab.cba.mit.edu/classes/864.17/text/pde_diff.pdf


