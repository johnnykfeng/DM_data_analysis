import numpy as np

# Tungsten PETG
DENSITY = 4.0 #g/cm^3 according to Prusa's article

def Intensity(x, I0, mu, offset):
    return I0 * np.exp(-mu * x) + offset

def HVL(mu):
    # ln(2)/mu
    return np.log(2)/mu 

def T_mean_to_mu(T_mean, x):
    return -np.log(T_mean)/x

def MFP(mu):
    return 1/mu

def mass_mu(mu, DENSITY):
    return mu/DENSITY

# testing git version control

