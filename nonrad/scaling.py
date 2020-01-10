import numpy as np
from itertools import groupby
from scipy import constants as const
from scipy.optimize import curve_fit
from numpy.polynomial.laguerre import laggauss
from pymatgen.io.vasp.outputs import Wavecar
try:
    from numba import njit
except ModuleNotFoundError:
    def njit(*args, **kwargs):
        def _njit(func):
            return func
        return _njit


def sommerfeld_parameter(T, Z, m_eff, eps0, method='Integrate'):
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
    method : str
        specify method for evaluating sommerfeld parameter ('Integrate' or
        'Analytic'). The default is recommended as the analytic equation may
        introduce significant errors for the repulsive case at high T.

    Returns
    -------
    float, np.array(dtype=float)
        sommerfeld factor evaluated at the given temperature
    """
    if Z == 0:
        return 1.

    if method.lower()[0] == 'i':
        kT = const.k * T
        m = m_eff * const.m_e
        eps = (4 * np.pi * const.epsilon_0) * eps0
        f = -2 * np.pi * Z * m * const.e**2 / const.hbar**2 / eps

        def s_k(k):
            return f / k / (1 - np.exp(-f / k))

        t = 0.
        x, w = laggauss(64)
        for ix, iw in zip(x, w):
            t += iw * np.sqrt(ix) * s_k(np.sqrt(2 * m * kT * ix) / const.hbar)
        return t / np.sum(w * np.sqrt(x))
    else:
        # that 4*pi from Gaussian units....
        theta_b = np.pi**2 * (m_eff * const.m_e) * const.e**4 / \
            (2 * const.k * const.hbar**2 * (eps0 * 4*np.pi*const.epsilon_0)**2)
        zthetaT = Z**2 * theta_b / T

        if Z < 0:
            return 4 * np.sqrt(zthetaT / np.pi)
        else:
            return (8 / np.sqrt(3)) * \
                zthetaT**(2/3) * np.exp(-3 * zthetaT**(1/3))


@njit(cache=True)
def find_charge_center(density, lattice):
    """
    Computes the center of the charge density.

    Parameters
    ----------
    density : np.array
        density of the wavecar returned by wavecar.fft_mesh
    lattice : np.array
        lattice to use to compute PBC

    Returns
    -------
    np.array
        position of the center in cartesian coordinates
    """
    avg = np.zeros(3)
    for i in range(density.shape[0]):
        for j in range(density.shape[1]):
            for k in range(density.shape[2]):
                avg += np.dot(np.array((i, j, k)) / np.array(density.shape),
                              lattice) * density[(i, j, k)]
    return avg / np.sum(density)


@njit(cache=True)
def distance_PBC(a, b, lattice):
    """
    Computes the distance between a and b on the lattice with periodic BCs.

    Parameters
    ----------
    a, b : np.array
        points in cartesian coordinates
    lattice : np.array
        lattice to use to compute PBC

    Returns
    -------
    float
        distance
    """
    min_dist = np.inf
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                R = np.dot(np.array((i, j, k), dtype=np.float64), lattice)
                dist = np.linalg.norm(a - b + R)
                min_dist = dist if dist < min_dist else min_dist
    return min_dist


@njit(cache=True)
def radial_distribution(density, point, lattice):
    """
    Computes the radial distribution.

    Computes the radial distribution of the density around the given point with
    the defined lattice.

    Parameters
    ----------
    density : np.array
        density of the wavecar returned by wavecar.fft_mesh
    point : np.array
        position of the defect in cartesian coordinates
    lattice : np.array
        lattice to use to compute PBC

    Returns
    -------
    r : np.array
        array of radii that the density corresponds to
    n : np.array
        array of densities at the corresponding radii
    """
    N = density.shape[0] * density.shape[1] * density.shape[2]
    r, n = (np.zeros(N), np.zeros(N))
    m = 0
    for i in range(density.shape[0]):
        for j in range(density.shape[1]):
            for k in range(density.shape[2]):
                r[m] = distance_PBC(np.dot(
                    np.array((i, j, k), dtype=np.float64) /
                    np.array(density.shape),
                    lattice
                ), point, lattice)
                n[m] = density[(i, j, k)]
                m += 1
    return r, n


def charged_supercell_scaling(wavecar_path, bulk_index, def_index=None,
                              def_coord=None, cutoff=0.02, limit=5., spin=0,
                              kpoint=1, fig=None, full_range=False):
    """
    Estimate the interaction between the defect and bulk wavefunction.

    This function estimates the interaction between the defect and bulk
    wavefunction due to spurious effects as a result of using a charged
    supercell. The radial distribution of the bulk wavefunction is compared to
    a perfectly homogenous wavefunction to estimate the scaling.

    Either def_index or def_coord must be specified.

    If you get wonky results with def_index, try using def_coord as there may
    be a problem with finding the defect position if the defect charge is at
    the boundary of the cell.

    Parameters
    ----------
    wavecar_path : str
        path to the WAVECAR file that contains the relevant wavefunctions
    def_index, bulk_index : int
        index of the defect and bulk wavefunctions in the WAVECAR file
    def_coord : np.array(dim=(3,))
        cartesian coordinates of defect position
    cutoff : float
        cutoff for determining zero slope regions
    limit : float
        upper limit for windowing procedure
    spin : int
        spin channel to read from (0 - up, 1 - down)
    kpoint : int
        kpoint to read from (defaults to the first kpoint)
    fig : matplotlib.figure.Figure
        optional figure object to plot diagnostic information (recommended)
    full_range : bool
        determines if full range of first plot is shown

    Returns
    -------
    float
        estimated scaling value to apply to the capture coefficient
    """
    if def_index is None and def_coord is None:
        raise ValueError('either def_index or def_coord must be specified')

    wavecar = Wavecar(wavecar_path)
    volume = np.dot(wavecar.a[0, :],
                    np.cross(wavecar.a[1, :], wavecar.a[2, :]))

    # compute relevant things
    if def_coord is None:
        psi_def = wavecar.fft_mesh(spin=spin, kpoint=kpoint-1,
                                   band=def_index-1)
        fft_psi_def = np.fft.ifftn(psi_def)
        den_def = np.abs(np.conj(fft_psi_def) * fft_psi_def) / \
            np.abs(np.vdot(fft_psi_def, fft_psi_def))
        def_coord = find_charge_center(den_def, wavecar.a)

    psi_bulk = wavecar.fft_mesh(spin=spin, kpoint=kpoint-1, band=bulk_index-1)
    fft_psi_bulk = np.fft.ifftn(psi_bulk)
    den_bulk = np.abs(np.conj(fft_psi_bulk) * fft_psi_bulk) / \
        np.abs(np.vdot(fft_psi_bulk, fft_psi_bulk))

    r, density = radial_distribution(den_bulk, def_coord, wavecar.a)

    # fitting procedure
    def f(x, alpha):
        return alpha * (4 * np.pi / 3 / volume) * x**3

    sind = np.argsort(r)
    R, int_den = (r[sind], np.cumsum(density[sind]))
    uppers = np.linspace(1., limit, 500)
    alphas = np.array([
        curve_fit(f, R[(R >= 0.) & (R <= u)],
                  int_den[(R >= 0.) & (R <= u)])[0][0]
        for u in uppers
    ])

    # find possible plateaus and return earliest one
    zeros = np.where(np.abs(np.gradient(alphas, uppers)) < cutoff)[0]
    plateaus = list(map(
        lambda x: (x[0], len(list(x[1]))),
        groupby(np.round(alphas[zeros], 2))
    ))

    if fig:
        tr = np.linspace(0., np.max(r), 100)
        ax = fig.subplots(1, 3)
        ax[0].scatter(R, int_den, s=10)
        ax[0].plot(tr, f(tr, 1.), color='r')
        ax[0].fill_between(tr, f(tr, 0.9), f(tr, 1.1),
                           facecolor='r', alpha=0.5)
        if not full_range:
            ax[0].set_xlim([-0.05 * limit, 1.05 * limit])
            ax[0].set_ylim([-0.05, 0.30])
        else:
            ax[0].set_ylim([-0.05, 1.05])
        ax[0].set_xlabel(r'radius [$\AA$]')
        ax[0].set_ylabel('cumulative charge density')
        ax[1].scatter(uppers, alphas, s=10)
        try:
            ax[1].axhline(y=plateaus[0][0], color='k', alpha=0.5)
        except IndexError:
            pass
        ax[1].set_xlabel(r'radius [$\AA$]')
        ax[1].set_ylabel(r'$\alpha$ [1]')
        ax[2].set_yscale('log')
        ax[2].scatter(uppers, np.abs(np.gradient(alphas, uppers)), s=10)
        ax[2].scatter(uppers[zeros],
                      np.abs(np.gradient(alphas, uppers))[zeros],
                      s=10, color='r', marker='.')
        ax[2].axhline(y=cutoff, color='k', alpha=0.5)
        ax[2].set_xlabel(r'radius [$\AA$]')
        ax[2].set_ylabel(r'$d \alpha / d r$ [1]')

    return plateaus[0][0]


def thermal_velocity(T, m_eff):
    """
    Calculates the thermal velocity at a given temperature

    Parameters
    ----------
    T : float, np.array(dtype=float)
        temperature in K
    m_eff : float
        effective mass in electron masses

    Returns
    -------
    float, np.array(dtype=float)
        thermal velocity at the given temperature in cm s^{-1}
    """
    return np.sqrt(3 * const.k * T / (m_eff * const.m_e)) * 1e2
