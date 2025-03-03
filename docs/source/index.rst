:html_theme.sidebar_secondary.remove: true

.. module:: meeg_simulator


Welcome meeg_simulator documentation
============================

meeg_simulator is a toolbox to simulate MEG/EEG data with pre-specified multivariate patterns representing experimental conditions of interest at specific time points. 
By simulating data with known, ground truth effects, it becomes possible to establish the valdity of the analysis pipelines developed for any event related experiment. With 
this toolbox, you can simulate data based on your very own experimental design and data set, and you can inject multivariate effects based on your own predictions. If you can detect
these effects in your simulated data, you know that your pipeline is working as it should. 


Installation
============

To install meeg_simulator, clone the meeg_simulator repository from
https://github.com/AlexLepauvre/meeg_simulator and run:

.. code-block:: bash

   pip install .

PyPI and conda releases are planned for the future.


.. toctree::
   :maxdepth: 2

   Tutorials <tutorials/index>
   API reference <reference/index>