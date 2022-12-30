# Copyright (c) Chris G. Van de Walle
# Distributed under the terms of the MIT License.

"""Core module for nonrad.

This module provides the main implementation to evaluate the nonradiative
capture coefficient from first-principles.
"""

import warnings
from typing import Union

import numpy as np
from scipy import constants as const
from scipy.interpolate import PchipInterpolator, interp1d

from nonrad.ccd import get_barrier_harmonic
from nonrad.constants import AMU2KG, ANGS2M, EV2J, HBAR

try:
    from numba import njit, vectorize

    @vectorize
    def herm_vec(x: float, n: int) -> float:
        """Wrap herm function."""
        return herm(x, n)
except ModuleNotFoundError:
    from numpy.polynomial.hermite import hermval

    def njit(*args, **kwargs):      # pylint: disable=W0613
        """Fake njit when numba can't be found."""
        def _njit(func):
            return func
        return _njit

    def herm_vec(x: float, n: int) -> float:
        """Wrap hermval function."""
        return hermval(x, [0.]*n + [1.])


factor = ANGS2M**2 * AMU2KG / HBAR / HBAR / EV2J
Factor2 = const.hbar / ANGS2M**2 / AMU2KG
Factor3 = 1 / HBAR

# for fast factorial calculations
LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype=np.double)

# range for computing overlaps in overlap_NM, precomputing once saves time
# note: -30 to 30 is an arbitrary region. It should be sufficient, but
# we should probably check this to be safe. 5000 is arbitrary also.
QQ = np.linspace(-30., 30., 5000)

# grid and weights for Hermite-Gauss quadrature in fast_overlap_NM
hgx, hgw = np.polynomial.hermite.hermgauss(256)


@njit(cache=True)
def fact(n: int) -> float:
    """Compute the factorial of n."""
    if n > 20:
        return LOOKUP_TABLE[-1] * \
            np.prod(np.array(list(range(21, n+1)), dtype=np.double))
    return LOOKUP_TABLE[n]


@njit(cache=True)
def herm(x: float, n: int) -> float:
    """Recursive definition of hermite polynomial."""
    if n == 0:
        return 1.
    if n == 1:
        return 2. * x

    y1 = 2. * x
    dy1 = 2.
    for i in range(2, n+1):
        yn = 2. * x * y1 - dy1
        dyn = 2. * i * y1
        y1 = yn
        dy1 = dyn
    return yn


@njit(cache=True)
def overlap_NM(
        DQ: float,
        w1: float,
        w2: float,
        n1: int,
        n2: int
) -> float:
    """Compute the overlap between two displaced harmonic oscillators.

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
    Hn1Q = herm_vec(np.sqrt(factor*w1)*(QQ-DQ), n1)
    Hn2Q = herm_vec(np.sqrt(factor*w2)*(QQ), n2)

    wfn1 = (factor*w1/np.pi)**(0.25)*(1./np.sqrt(2.**n1*fact(n1))) * \
        Hn1Q*np.exp(-(factor*w1)*(QQ-DQ)**2/2.)
    wfn2 = (factor*w2/np.pi)**(0.25)*(1./np.sqrt(2.**n2*fact(n2))) * \
        Hn2Q*np.exp(-(factor*w2)*QQ**2/2.)

    return np.trapz(wfn2*wfn1, x=QQ)


@njit(cache=True)
def fast_overlap_NM(
        dQ: float,
        w1: float,
        w2: float,
        n1: int,
        n2: int
) -> float:
    """Compute the overlap between two displaced harmonic oscillators.

    This function computes the overlap integral between two harmonic
    oscillators with frequencies w1, w2 that are displaced by dQ for the
    quantum numbers n1, n2. The integral is computed using Hermite-Gauss
    quadrature; this method requires an order of magnitude less grid points
    than the trapezoid method.

    Parameters
    ----------
    dQ : float
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
    fw1, fw2 = (factor * w1, factor * w2)
    x2Q = np.sqrt(2. / (fw1 + fw2))

    def _g(Q):
        h1 = herm_vec(np.sqrt(fw1)*(Q-dQ), n1) \
            / np.sqrt(2.**n1 * fact(n1)) * (fw1/np.pi)**(0.25)
        h2 = herm_vec(np.sqrt(fw2)*Q, n2) \
            / np.sqrt(2.**n2 * fact(n2)) * (fw2/np.pi)**(0.25)
        return np.exp(fw1*dQ*(Q-dQ/2.)) * h1 * h2

    return x2Q * np.sum(hgw * _g(x2Q*hgx))


@njit(cache=True)
def analytic_overlap_NM(
        DQ: float,
        w1: float,
        w2: float,
        n1: int,
        n2: int
) -> float:
    """Compute the overlap between two displaced harmonic oscillators.

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
    w = np.double(w1 * w2 / (w1 + w2))
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
            l = 2 * lx + l2     # noqa: E741
            Pr2 = (fact(n1) * fact(n2))**0.5 / \
                (fact(k)*fact(l)*fact(k1-kx)*fact(l1-lx)) * \
                2**((k + l - n2 - n1) / 2)
            Pr3 = (sinfi**k)*(cosfi**l)
            # f = hermval(rho, [0.]*(k+l) + [1.])
            f = herm(np.float64(rho), k+l)
            Ix = Ix + Pr1*Pr2*Pr3*f
    return Ix


def get_C(
        dQ: float,
        dE: float,
        wi: float,
        wf: float,
        Wif: float,
        volume: float,
        g: int = 1,
        T: Union[float, np.ndarray] = 300.,
        sigma: Union[str, float] = 'pchip',
        occ_tol: float = 1e-5,
        overlap_method: str = 'HermiteGauss'
) -> Union[float, np.ndarray]:
    """Compute the nonradiative capture coefficient.

    This function computes the nonradiative capture coefficient following the
    methodology of A. Alkauskas et al., Phys. Rev. B 90, 075202 (2014). The
    resulting capture coefficient is unscaled [See Eq. (22) of the above
    reference]. Our code assumes harmonic potential energy surfaces.

    Parameters
    ----------
    dQ : float
        displacement between harmonic oscillators in amu^{1/2} Angstrom
    dE : float
        energy offset between the two harmonic oscillators
    wi, wf : float
        frequencies of the harmonic oscillators in eV
    Wif : float
        electron-phonon coupling matrix element in eV amu^{-1/2} Angstrom^{-1}
    volume : float
        volume of the supercell in Å^3
    g : int
        degeneracy factor of the final state
    T : float, np.array(dtype=float)
        temperature or a np.array of temperatures in K
    sigma : 'pchip', 'cubic', or float
        smearing parameter in eV for replacement of the delta functions with
        gaussians. A value of 'pchip' or 'cubic' corresponds to interpolation
        instead of gaussian smearing, utilizing PCHIP or cubic spline
        interpolaton. PCHIP is preferred to cubic spline as cubic spline can
        result in negative values when small rates are found. The default is
        'pchip' and is recommended for improved accuracy.
    occ_tol : float
        criteria to determine the maximum quantum number for overlaps based on
        the Bose weights
    overlap_method : str
        method for evaluating the overlaps (only the first letter is checked)
        allowed values => ['Analytic', 'Integral', 'HermiteGauss']

    Returns
    -------
    float, np.array(dtype=float)
        resulting capture coefficient (unscaled) in cm^3 s^{-1}
    """
    volume *= (1e-8)**3
    kT = (const.k / const.e) * T    # [(J / K) * (eV / J)] * K = eV
    Z = 1. / (1 - np.exp(-wi / kT))

    Ni, Nf = (17, 50)   # default values
    tNi = np.ceil(-np.max(kT) * np.log(occ_tol) / wi).astype(int)
    if tNi > Ni:
        Ni = tNi
    tNf = np.ceil((dE + Ni*wi) / wf).astype(int)
    if tNf > Nf:
        Nf = tNf

    if (Eb := get_barrier_harmonic(dQ, dE, wi, wf)) is not None \
            and (N_barrier := np.ceil(Eb / wi).astype(int)) > Ni:
        warnings.warn('Number of initial phonon states included does not '
                      f'reach the barrier height ({N_barrier} > {Ni}). '
                      'You may want to test the sensitivity to occ_tol.',
                      RuntimeWarning, stacklevel=2)

    # warn if there are large values, can be ignored if you're confident
    if Ni > 150 or Nf > 150:
        warnings.warn(f'Large value for Ni, Nf encountered: ({Ni}, {Nf})',
                      RuntimeWarning, stacklevel=2)

    # precompute values of the overlap
    ovl = np.zeros((Ni, Nf), dtype=np.longdouble)
    for m in np.arange(Ni):
        for n in np.arange(Nf):
            if overlap_method.lower()[0] == 'a':
                ovl[m, n] = analytic_overlap_NM(dQ, wi, wf, m, n)
            elif overlap_method.lower()[0] == 'i':
                ovl[m, n] = overlap_NM(dQ, wi, wf, m, n)
            elif overlap_method.lower()[0] == 'h':
                ovl[m, n] = fast_overlap_NM(dQ, wi, wf, m, n)
            else:
                raise ValueError(f'Invalid overlap method: {overlap_method}')

    t = np.linspace(-Ni*wi, Nf*wf, 5000)
    R = 0.
    for m in np.arange(Ni-1):
        weight_m = np.exp(-m * wi / kT) / Z
        if isinstance(sigma, str):
            # interpolation to replace delta functions
            E, matels = (np.zeros(Nf), np.zeros(Nf))
            for n in np.arange(Nf):
                if m == 0:
                    matel = np.sqrt(Factor2 / 2 / wi) * ovl[1, n] + \
                        np.sqrt(Factor3) * dQ * ovl[0, n]
                else:
                    matel = np.sqrt((m+1) * Factor2 / 2 / wi) * ovl[m+1, n] + \
                        np.sqrt(m * Factor2 / 2 / wi) * ovl[m-1, n] + \
                        np.sqrt(Factor3) * dQ * ovl[m, n]
                E[n] = n*wf - m*wi
                matels[n] = np.abs(np.conj(matel) * matel)
            if sigma[0].lower() == 'c':
                f = interp1d(E, matels, kind='cubic', bounds_error=False,
                             fill_value=0.)
            else:
                f = PchipInterpolator(E, matels, extrapolate=False)
            R = R + weight_m * (f(dE) * np.sum(matels)
                                / np.trapz(np.nan_to_num(f(t)), x=t))
        else:
            # gaussian smearing with given sigma to replace delta functions
            for n in np.arange(Nf):
                # energy conservation delta function
                delta = np.exp(-(dE+m*wi-n*wf)**2/(2.0*sigma**2)) / \
                    (sigma*np.sqrt(2.0*np.pi))
                if m == 0:
                    matel = np.sqrt(Factor2 / 2 / wi) * ovl[1, n] + \
                        np.sqrt(Factor3) * dQ * ovl[0, n]
                else:
                    matel = np.sqrt((m+1) * Factor2 / 2 / wi) * ovl[m+1, n] + \
                        np.sqrt(m * Factor2 / 2 / wi) * ovl[m-1, n] + \
                        np.sqrt(Factor3) * dQ * ovl[m, n]
                R = R + weight_m * delta * np.abs(np.conj(matel) * matel)
    return 2 * np.pi * g * Wif**2 * volume * R
