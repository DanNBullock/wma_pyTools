# White Matter Anatomy Python Tools (wma_PyTools)

## Overview description

This repository contains a number of python code tools which can be used to perform anatomically informed segmentations and analyze the outputs.  Initially based on the matlab-based [wma_tools toolset](https://github.com/DanNBullock/wma_tools)

## Author, funding sources, references

### Authors
- [Dan Bullock](https://github.com/DanNBullock/) ([bullo092@umn.edu](mailto:bullo092@umn.edu))

### PI
- [Sarah Heilbronner](https://med.umn.edu/bio/department-of-neuroscience/sarah-heilbronner) ([heilb028@umn.edu](mailto:heilb028@umn.edu))

### Funding
[![NIH-NIBIB-1T32EB031512-01](https://img.shields.io/badge/NIH_NIBIB-1T32EB031512--01-blue.svg)](https://reporter.nih.gov/project-details/10205698)
[![NIMH-5P50MH119569-02](https://img.shields.io/badge/NIMH-5P50MH119569--02-blue.svg)](https://reporter.nih.gov/project-details/10123009)
[![NIMH-5R01MH118257-04](https://img.shields.io/badge/NIMH-5R01MH118257--04-blue.svg)](https://reporter.nih.gov/project-details/10122991)
[![NIDA-1P30DA048742-01A1](https://img.shields.io/badge/NIDA-1P30DA048742--01A1-blue.svg)](https://reporter.nih.gov/project-details/10025457)

* [Dan Bullock](https://github.com/DanNBullock/)'s work is supported by the following sources:
    - The University of Minnesota’s Neuroimaging Postdoctoral Fellowship Program, College of Science and Engineering's, and the Medical School's NIH funded T32 Neuroimaging Training Grant. NOGA: [1T32EB031512-01](https://reporter.nih.gov/project-details/10205698) 

- [Sarah Heilbronner](https://med.umn.edu/bio/department-of-neuroscience/sarah-heilbronner)'s work is supported by the following sources:
    - The University of Minnesota’s Neurosciences' and Medical School's NIMH grant for the investigation of "Neural Basis of Psychopathology, Addictions and Sleep Disorders Study Section[NPAS]". NOGA: [5P50MH119569-02-04](https://reporter.nih.gov/project-details/10123009) 
    - The University of Minnesota’s Neurosciences' Translational Neurophysiology grant. NOGA: [5R01MH118257-04](https://reporter.nih.gov/project-details/10122991)
    - The University of Minnesota’s Neurosciences' Addiction Connectome grant. NOGA: [1P30DA048742-01A1](https://reporter.nih.gov/project-details/10025457)

## Development note
Work to better structure and document this code is ongoing.  Please feel free to create issues to provoke clarity in documentation, or to suggest alterations to documentation or code more directly. Furthermore, it is acutely understood that many of the functionalities in this package may be redundant implementations of (likely more robust and reliable) functionalities from other packages. The identification of such instances (and their other-package correspondances) would be **greatly appreciated**. 

## Package/Module overview

The wma_PyTools code collection is predominantly housed within the wmaPyTools directory of this repository (**failedCode** contains code implementations that may or may not be functional, but which were somehow determined to be ineffectual relative to their intended uses/applications, while **testsOrDemos** contains code that benchmarks or sanity-checks code compositions from wmaPyTools).

Currently, there are five primary subdivisions into which functions have been organized:

- roiTools
- visTools
- segmentationTools
- streamlineTools
- analysisTools

### roiTools
These functions typically relate to the creation, modification, utilization of NiFTI based ROIs and masks. For example, these functions can be used to extract ROIs from a volumetric atlas/parcellation, inflate or modify such ROIs, or generate new planar ROIs based on the borders of other ROIs.

### visTools
These functions typically relate to the creation of plots or visualizations of streamline tractography objects (e.g. a plot of a "tract"), NiFTI-based entity (e.g. a gif visualization of volumetric streamline density), or quantative analyses (facilitated by analyisTools) thereof.

### analysisTools
These functions typically relate to the performance of quantative analyses of streamline or NiFTI based data objects. Such functions typically faciltate the performance of more complex operations (e.g. the computation of streamline distance traversal) or are used to generate data for visualizations/plots (e.g. those found in visTools)

### streamlineTools
These functions typically relate to operations performed on or with streamlines.

### segmentationTools
These functions typically invovle the assesment or imposition of quantative criteria relative to an input collection of streamlines.  Such criteria are typically posed or instantiated in relation to ROIs derived from roiTools operations.

## Required packages
(non-exhaustive listing of directly called packages)

- [DIPY](https://github.com/dipy/dipy)
- [scipy]
- [nilearn]
- [nibabel]
- [matplotlib]
- [pandas]
- [mpl_toolkits]
- [numpy]
- [PIL]
...
