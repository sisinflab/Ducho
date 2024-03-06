Installation
----------------

Depending on where you are running Ducho, you might need to first clone this repo and install the necessary packages.

If you are running Ducho on your local machine or Google Colab, you first need to clone this repository:

.. code-block:: bash

    $ git clone https://github.com/sisinflab/Ducho.git


Then, install the needed dependencies through pip:

.. code-block:: bash

    $ pip install -r requirements.txt # Local
    $ pip install -r requirements_colab.txt # Google Colab


P.S. Since Google Colab already comes with almost all necessary packages, you just need to install very few missing ones.

Note that these two steps are not necessary for the docker version because the image already comes with the suitable environment.
