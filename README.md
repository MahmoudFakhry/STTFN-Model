# What is a Spatiotemporal Fusion Network? (STTFN)

_Our implementation is based on the works of Zhixianh Yin, Penghai Wu, and Giles M. Foody in "Spatiotemporal Fusion of Land Surface Temperature Based on a Convolutional Neural Network"_

The specific research paper, including the model architecture, can be found here: https://ieeexplore.ieee.org/document/9115890

A Spatiotemporal Fusion Network is a multiscale fusion-based convolutional neural network utilized to build nonlinear relationships between input and output images -- in this case, Land Surface Temperature (LST) satellite imaging. By utilizing two convolutional neural networks to predict a forward sequence and a backward sequence, we can predict a "middle" sequence, filling in possible missing or damaged LST satellite imaging.

In remote sensing, MODIS satellite data is considered lower resolution, but common -- while Landsat satellite data is higher resolution, but rarer. Utilizing this STTFN method, we aim to use the context of surrounding Landsat and MODIS imagery to infer Landsat-quality imaging for days where there would typically be none.

We implemented the STTFN research paper, which is provided in the .ipnyb file, and trained on Oregon State University's HPC Clusters. For performance metrics, we used Root Mean Squared Error (RMSE) and Structural Similarity (SSIM).
The .ipnyb file also includes an implementation of a "comparison model", the STARFM model (original implementation and repository can be found at https://github.com/nmileva/starfm4py).

_An example of our implemented STTFN results can be seen below._

![image](https://github.com/Todd-C-Goldfarb/STTFN-OSU/assets/132838573/1922a493-65c4-408d-bdfc-bd36205602a0) 

_Pictured above is the "target MODIS image"._

![image](https://github.com/Todd-C-Goldfarb/STTFN-OSU/assets/132838573/5817bd16-d459-4eaa-972c-58cab6ca50b9)

_The MODIS image has been converted into the "inferred Landsat-Quality" model output._

_The model output's SSIM: 0.9968653789596235_

_The model output's RMSE: 1.726944568447375_

![image](https://github.com/Todd-C-Goldfarb/STTFN-OSU/assets/132838573/c1670817-a0b3-401e-9fe1-f51ec135cb50)

_The "Goal" Landsat (Notice the similarity, and how the Model Output ignores the Data's artifacts)_

## Using the .ipnyb

We encourage anyone looking to try out the .ipnyb file, though if you don't have access to University-level HPC Clusters, then the L4 GPU on the Google Colab should work just fine for this purpose.

Provide a named area of interest compatible with the Microsoft Planetary Computer connectors, and test your area of interest.

## Disclaimer
This repository, and it's contents, are the result of the 2023-2024 STTFN Senior Project at Oregon State University with constellr gmbh.

This repository has a strict license to ensure read-only rights, for educational and research purposes. Check the LICENSE.md for more.

Team Members:

Todd Goldfarb (tcgoldfarb@gmail.com)

Joseph Balaty (balatyj@oregonstate.edu)

Jarod Lokrantz (lokrantj@oregonstate.edu)

Mahmoud Fahkry (fakhryk@oregonstate.edu)


