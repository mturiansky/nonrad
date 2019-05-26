import numpy as np
from scipy import constants as const


def sommerfeld_parameter(T, Z, m_eff, eps0):
    """
    Compute the sommerfeld parameter

    Computes the sommerfeld parameter at a given temperature using the
    definitions in R. Pässler et al., phys. stat. sol. (b) 78, 625 (1976). We
    assume that theta_{b,i}(T) ~ T.

    Parameters
    ----------
    T : float, np.array(dtype=float)
        temperature in K
    Z : int
        Z = Q / q where Q is the charge of the defect and q is the charge of
        the carrier. Z < 0 corresponds to attractive centers and Z > 0
        corresponds to repulsive centers
    m_eff : float
        effective mass of the carrier in units of m_e (electron mass)
    eps0 : float
        static dielectric constant

    Returns
    -------
    float, np.array(dtype=float)
        sommerfeld factor evaluated at the given temperature
    """
    # that 4*pi from Gaussian units....
    theta_b = np.pi**2 * (m_eff * const.m_e) * const.e**4 / \
        (2 * const.k * const.hbar**2 * (eps0 * 4*np.pi*const.epsilon_0)**2)
    zthetaT = Z**2 * theta_b / T

    if Z < 0:
        return 4 * np.sqrt(zthetaT / np.pi)
    elif Z > 0:
        return (8 / np.sqrt(3)) * zthetaT**(2/3) * np.exp(-3 * zthetaT**(1/3))
    else:
        return 1.
