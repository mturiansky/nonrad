=============
Tips & Tricks
=============

Below we compile several useful tips and tricks related to running `nonrad`.


1. Known bug in VASP 5.4.4
--------------------------

In version 5.4.4 of `VASP`, a bug is present in the code, which results in potential errors in the computed electron-phonon coupling.
The easy solution is to use version :math:`\ge` 6.0.0 of `VASP`, for which the bug is already fixed.
Otherwise, the `VASP` source code can be modified to fix the bug, as described at `here <https://github.com/mturiansky/nonrad/issues/2#issuecomment-1084963299>`_.

2. Systems with large barriers
------------------------------

Many of the default parameters and choices in the design of `nonrad` are based around conventional recombination or trapping centers (generally, :math:`C > 10^{-13}`).
If you apply the code to more unusual cases, care needs to be taken to ensure convergence.
For example, one may try to calculate the capture coefficient for a system with a large semiclassical barrier to recombination (the crossing point between the potential energy surfaces).
If the barrier energy is too large, then the calculation may be sensitive to the ``occ_tol`` parameter.
``occ_tol`` is used to determine, along with the temperature and phonon frequency, the highest phonon quantum number of the initial state included in the calculation.
When the barrier is large and the temperature is high, higher quantum numbers may be important (i.e., :math:`m_{\rm max} \hbar\Omega_i > E_b`, where :math:`E_b` is the barrier energy, is desired).
Of course, the conventional wisdom would tell you that a system with such a large barrier can be neglected, so one does not even need to explicitly perform the calculation.
