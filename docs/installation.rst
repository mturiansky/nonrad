============
Installation
============

`nonrad` may be installed through pip from PyPI,

.. code-block:: sh

    pip install nonrad

or directly through github,

.. code-block:: sh

    pip install git+https://github.com/mturiansky/nonrad


Going faster
============

This code utilizes `numba` to speed up calculations, and there are various ways to improve the performance of `numba`.

SVML
----

On Intel processors, the short vector math library (SVML) can be enabled to speed up certain operations.
The runtime libraries from Intel are required for this.
On a `conda` installation, they should already be installed in the package `icc_rt`.
The `icc_rt` package is also available through pip

.. code-block:: sh

    pip install icc_rt

However, you will likely need to add your virtual environment to the library path:

.. code-block:: sh

    export LD_LIBRARY_PATH=/path/to/.virtualenvs/env_name/lib/:$LD_LIBRARY_PATH

This can be added to your `activate` script in your virtual environment (``/path/to/.virtualenvs/env_name/bin/activate``) to make the change persistent.
To check if the installation worked, run ``numba -s``; the output should include

.. code-block::

   ...

   __SVML Information__
    SVML State, config.USING_SVML                 : True
    SVML Library Loaded                           : True
    llvmlite Using SVML Patched LLVM              : True
    SVML Operational                              : True

   ...

Numba Enviornment Variables
---------------------------

There are several environment variables for `numba` that can be enabled and may improve performance.
If your machine has AVX instructions, we recommend enabling it with:

.. code-block:: sh

   export NUMBA_ENABLE_AVX=1

The full list of `numba` environment variables is available `here <https://numba.pydata.org/numba-doc/latest/reference/envvars.html#compilation-options>`_.
