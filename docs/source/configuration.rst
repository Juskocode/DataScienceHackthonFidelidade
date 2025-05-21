Configuration
==============

You can install **xsell_dental_exemplo** with ``pip`` or from source.

.. note::
    **xsell_dental_exemplo** only supports Python 3.7 or above so make you sure you have it installed before proceeding.

Install from Source
-------------------

First, ensure that you have the latest pip version to avoid dependency errors::

   pip install --upgrade pip


To install caa from source, clone the repository from `gitlab
<https://dev.azure.com/caa-fid/Code%20Stack/_git/caa>`_::

    git clone https://dev.azure.com/caa-fid/Code%20Stack/_git/caa
    cd caa
    pip install .

You can view the list of all dependencies within the ``install_requires`` field
of ``setup.py``.

Run Tests
-----------

Test caa with ``pytest``. If you don't have ``pytest`` installed run::

    pip install pytest

Then to run all tests just run::

    pytest .
