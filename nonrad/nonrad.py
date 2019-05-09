import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.special import factorial
from scipy.interpolate import interp1d
from scipy import constants as const

hbar = 4.135667662e-15 / 2 / np.pi      # in units of eV.s
HBAR = const.hbar / const.e     # in units of eV.s
eV2J = 1.60217662e-19                   # 1 eV in Joules
EV2J = const.e                      # 1 eV in Joules
amu2kg = 1.660539040e-27                # atomic mass unit in kg
AMU2KG = const.physical_constants['atomic mass constant'][0]
angs2m = 1e-10                          # angstrom in meters
ANGS2M = 1e-10                          # angstrom in meters

factor = 1/hbar/hbar/eV2J*amu2kg*angs2m*angs2m
Factor2 = 6.351E12                      # hbar/1E-20/amu
Factor3 = 1.52E15                       # e/hbar


def fact(n):
    """ wrapper for scaipy.special.factorial with exact=True """
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


def get_C(DQ, DE, w1, w2, V, Omega, g=1, T=300, sigma=None,
          overlap_method='analytic'):
    """

    Parameters
    ----------
    DQ : float
        displacement between harmonic oscillators in amu^{1/2} Angstrom
    DE : float
        energy offset between the two harmonic oscillators
    w1, w2 : float
        frequencies of the harmonic oscillators in eV
    V : float
        electron-phonon coupling matrix element in eV amu^{-1/2} Angstrom^{-1}
    Omega : float
        volume of the supercell in m^3
    g : int
        degeneracy factor of the final state
    T : float, np.array(dtype=float)
        temperature or a np.array of temperatures in K
    sigma : None or float
        smearing parameter in eV for replacement of the delta functions with
        gaussians. A value of None corresponds to interpolation instead of
        gaussian smearing. The default is None and is recommended for improved
        accuracy.
    overlap_method : str
        method for evaluating the overlaps (only the first letter is checked)
        allowed values => ['Analytic', 'Integral']

    Returns
    -------
    np.longdouble
        overlap of the two harmonic oscillator wavefunctions
    """
    kT = (const.k / const.e) * T    # [(J / K) * (eV / J)] * K = eV
    Z = 1. / (1 - np.exp(-w1 / kT))

    # these should be checked for consistency with temperature
    Ni, Nf = (17, 50)

    # precompute values of the overlap
    ovl = np.zeros((Ni, Nf), dtype=np.longdouble)
    for m in np.arange(Ni):
        for n in np.arange(Nf):
            if overlap_method.lower()[0] == 'a':
                ovl[m, n] = analytic_overlap_NM(DQ, w1, w2, m, n)
            elif overlap_method.lower()[0] == 'i':
                ovl[m, n] = overlap_NM(DQ, w1, w2, m, n)

    t = np.linspace(0, Nf*w2, 1000)
    R = 0.
    for m in np.arange(Ni-1):
        weight_m = np.exp(-m * w1 / kT) / Z
        if sigma is None:
            # interpolation to replace delta functions
            E, matels = (np.zeros(Nf), np.zeros(Nf))
            for n in np.arange(Nf):
                if m == 0:
                    matel = np.sqrt(Factor2 / 2 / w1) * ovl[1, n] + \
                        np.sqrt(Factor3) * DQ * ovl[0, n]
                else:
                    matel = np.sqrt((m+1) * Factor2 / 2 / w1) * ovl[m+1, n] + \
                        np.sqrt(m * Factor2 / 2 / w1) * ovl[m-1, n] + \
                        np.sqrt(Factor3) * DQ * ovl[m, n]
                E[n] = n*w2 - m*w1
                matels[n] = np.abs(np.conj(matel) * matel)
            f = interp1d(E, matels, kind='cubic', bounds_error=False,
                         fill_value=0.)
            R = R + weight_m * (f(DE) * np.sum(matels) / np.trapz(f(t), x=t))
        else:
            # gaussian smearing with given sigma to replace delta functions
            for n in np.arange(Nf):
                # energy conservation delta function
                delta = np.exp(-(DE+m*w1-n*w2)**2/(2.0*sigma**2)) / \
                    (sigma*np.sqrt(2.0*np.pi))
                if m == 0:
                    matel = np.sqrt(Factor2 / 2 / w1) * ovl[1, n] + \
                        np.sqrt(Factor3) * DQ * ovl[0, n]
                else:
                    matel = np.sqrt((m+1) * Factor2 / 2 / w1) * ovl[m+1, n] + \
                        np.sqrt(m * Factor2 / 2 / w1) * ovl[m-1, n] + \
                        np.sqrt(Factor3) * DQ * ovl[m, n]
                R = R + weight_m * delta * np.abs(np.conj(matel) * matel)
    return 2 * np.pi * g * V**2 * Omega * R
