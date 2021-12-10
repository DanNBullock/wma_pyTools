# White Matter Anatomy Python Tools (wma_PyTools)

## Overview description

This repository contains a number of python code tools which can be used to perform anatomically informed segmentations and analyze the outputs.  Initially based on the matlab-based [wma_tools toolset](https://github.com/DanNBullock/wma_tools)

## Author, funding sources, references

### Authors
- [Dan Bullock](https://github.com/DanNBullock/) ([bullo092@umn.edu](mailto:bullo092@umn.edu))

### PI
- [Sarah Heilbronner](https://med.umn.edu/bio/department-of-neuroscience/sarah-heilbronner) ([heilb028@umn.edu](mailto:heilb028@umn.edu))

### Funding

[This section needs expansion. You can help by adding to it]

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
