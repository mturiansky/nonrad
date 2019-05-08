import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import factorial, erf
from scipy.interpolate import interp1d
from scipy import constants

hbar = 4.135667662e-15 / 2 / np.pi      # in units of eV.s
HBAR = constants.hbar / constants.e     # in units of eV.s
eV2J = 1.60217662e-19                   # 1 eV in Joules
EV2J = constants.e                      # 1 eV in Joules
amu2kg = 1.660539040e-27                # atomic mass unit in kg
AMU2KG = constants.physical_constants['atomic mass constant'][0]
angs2m = 1e-10                          # angstrom in meters
ANGS2M = 1e-10                          # angstrom in meters

factor = 1/hbar/hbar/eV2J*amu2kg*angs2m*angs2m


def fact(n):
    return factorial(n, exact=True)


def overlap_NM(DQ, w1, w2, n1, n2):
    """
    Compute the overlap between two displaced harmonic oscillators.

    This function computes the overlap integral between two harmonic
    oscillators with frequencies w1, w2 that are displaced by DQ for the
    quantum numbers n1, n2. The integral is computed using the trapezoid
    method and the analytic form for the wavefunctions.

    Parameters
    ----------
    DQ : float
        displacement between harmonic oscillators in amu^{1/2} Angstrom
    w1, w2 : float
        frequencies of the harmonic oscillators in eV
    n1, n2 : integer
        quantum number of the overlap integral to calculate

    Returns
    -------
    np.longdouble
        overlap of the two harmonic oscillator wavefunctions
    """
    # note: -30 to 30 is an arbitrary region. It should be sufficient, but
    # we should probably check this to be safe. 1000 is arbitrary also.
    QQ = np.linspace(-30, 30, 1000, dtype=np.longdouble)

    Hn1Q = hermval(np.sqrt(factor*w1)*(QQ-DQ), [0.]*n1 + [1.])
    Hn2Q = hermval(np.sqrt(factor*w2)*(QQ), [0.]*n2 + [1.])

    wfn1 = (factor*w1/np.pi)**(0.25)*(1./np.sqrt(2.**n1*fact(n1))) * \
        Hn1Q*np.exp(-(factor*w1)*(QQ-DQ)**2/2.)
    wfn2 = (factor*w2/np.pi)**(0.25)*(1./np.sqrt(2.**n2*fact(n2))) * \
        Hn2Q*np.exp(-(factor*w2)*QQ**2/2.)

    return np.trapz(wfn2*wfn1, x=QQ)


def analytic_overlap_NM(DQ, w1, w2, n1, n2):
    """
    Compute the overlap between two displaced harmonic oscillators.

    This function computes the overlap integral between two harmonic
    oscillators with frequencies w1, w2 that are displaced by DQ for the
    quantum numbers n1, n2. The integral is computed using an analytic formula
    for the overlap of two displaced harmonic oscillators. The method comes
    from B.P. Zapol, Chem. Phys. Lett. 93, 549 (1982).

    Parameters
    ----------
    DQ : float
        displacement between harmonic oscillators in amu^{1/2} Angstrom
    w1, w2 : float
        frequencies of the harmonic oscillators in eV
    n1, n2 : integer
        quantum number of the overlap integral to calculate

    Returns
    -------
    np.longdouble
        overlap of the two harmonic oscillator wavefunctions
    """
    w = np.longdouble(w1 * w2 / (w1 + w2))
    rho = np.sqrt(factor) * np.sqrt(w / 2) * DQ
    sinfi = np.sqrt(w1) / np.sqrt(w1 + w2)
    cosfi = np.sqrt(w2) / np.sqrt(w1 + w2)

    Pr1 = (-1)**n1 * np.sqrt(2 * cosfi * sinfi) * np.exp(-rho**2)
    Ix = 0.
    k1 = n2 // 2
    k2 = n2 % 2
    l1 = n1 // 2
    l2 = n1 % 2
    for kx in range(k1+1):
        for lx in range(l1+1):
            k = 2 * kx + k2
            l = 2 * lx + l2             # noqa: E741
            Pr2 = (fact(n1) * fact(n2))**0.5 / \
                (fact(k)*fact(l)*fact(k1-kx)*fact(l1-lx)) * \
                2**((k + l - n2 - n1) / 2)
            Pr3 = (sinfi**k)*(cosfi**l)
            f = hermval(rho, [0.]*(k+l) + [1.])
            Ix = Ix + Pr1*Pr2*Pr3*f
    return Ix
