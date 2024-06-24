## Contacts
If you have any questions, or struggle with the code, please feel free to contact us at any of the emails below or create a Github Issue, which we will look to resolve.

Team Members:

Todd Goldfarb (tcgoldfarb@gmail.com)

Joseph Balaty (balatyj@oregonstate.edu)

Jarod Lokrantz (lokrantj@oregonstate.edu)

Mahmoud Fahkry (fakhryk@oregonstate.edu)

# Spatiotemporal Temperature Fusion Network & StarFM
### Using machine learning to turn unreadable or missing satellite images into usable and rich data.

![image](https://github.com/Todd-C-Goldfarb/STTFN-OSU/assets/132838573/9f5d6f1d-3323-42ad-8854-05959b15420c)

_An example implementation of the STTFN._

This repository, and it's contents, are the result of the 2023-2024 STTFN Senior Project at Oregon State University with constellr gmbh.

This repository has a strict license to ensure read-only rights, for educational and research purposes. Check the LICENSE.md for more.

# What is a Spatiotemporal Temperature Fusion Network? (STTFN)

_Our implementation is based on the works of Zhixianh Yin, Penghai Wu, and Giles M. Foody in "Spatiotemporal Fusion of Land Surface Temperature Based on a Convolutional Neural Network"_

The specific research paper, including the model architecture, can be found here: https://ieeexplore.ieee.org/document/9115890

A Spatiotemporal Temperature Fusion Network (STTFN) is a multiscale fusion-based convolutional neural network utilized to build nonlinear relationships between input and output images -- in this case, Land Surface Temperature (LST) satellite imaging. By utilizing two convolutional neural networks to predict a forward sequence and a backward sequence, we can predict a "middle" sequence, filling in possible missing or damaged LST satellite imaging.

In remote sensing, MODIS satellite data is considered lower resolution, but common -- while Landsat satellite data is higher resolution, but rarer. Utilizing this STTFN method, we aim to use the context of surrounding Landsat and MODIS imagery to infer Landsat-quality imaging for days where there would typically be none. For later descriptions in this document, Landsat imagery will be referred to as L# (with the # dictating the timestamp in relation to the prediction image), and MODIS will be referred to as M#, with the same rules.

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

# The Underlying Architecture

The general architecture of the Spatiotemporal Temperature Fusion Network consists of three primary factors:

1. A trained forward convolutional neural network.
2. A trained backward convolutional neural network.
3. An STC-Weighting function to average the results of the above-mentioned CNNs for the final prediction.

The architecture of the model, as well as the training flow, is specified below:

![image](https://github.com/Todd-C-Goldfarb/STTFN-OSU/assets/132838573/559e259a-41e0-48f8-bac0-49cdc79e5934)

_Source: "Spatiotemporal Fusion of Land Surface Temperature Based on a Convolutional Neural Network" (https://ieeexplore.ieee.org/document/9115890)_

## The Convolutional Neural Network
Both the forward trained and the backward trained neural network have the exact same shape and architecture, their difference comes from the training data provided.
For the forward trained network, the model is fed days/timestamps _leading up to the prediction-needed/missing/damaged image_, while the backward trained network is fed days/timesamps _counting down from the prediction-needed/missing/damaged image_. In this way, there are two models -- one essentially trying to predict the "next sequence", and another trying to predict the "precursor".

Each Convolutional Neural Network has three defining networks that process and learn from the images provided -- a _Super-Resolution Net_, an _Integration Net_ and _Extraction Net_.

The specific architecture for the Convolutional Neural Network can be seen below:

![image](https://github.com/Todd-C-Goldfarb/STTFN-OSU/assets/132838573/73be2267-f28a-4604-a65f-af79738c0e46)

_Source: "Spatiotemporal Fusion of Land Surface Temperature Based on a Convolutional Neural Network" (https://ieeexplore.ieee.org/document/9115890)_

## The STC-Weighting Function

The final part of the STTFN is to weigh the average of the forward trained CNN and the backward trained CNN for an accurate upscaling of the given MODIS image.

The STC-Weighting function is the function that calculates the weights/preference given over a certain CNN output based on it's accuracy to the given MODIS image the STTFN is trying to upscale.
The formula is specified below:

![image](https://github.com/Todd-C-Goldfarb/STTFN-OSU/assets/132838573/52354d7a-8e28-48ed-944c-6a28e9ac5518)

![image](https://github.com/Todd-C-Goldfarb/STTFN-OSU/assets/132838573/466503eb-71de-44dd-b327-fb53c72e1a22)


_Source: "Spatiotemporal Fusion of Land Surface Temperature Based on a Convolutional Neural Network" (https://ieeexplore.ieee.org/document/9115890)_

_i_ denotes the timestamp in relation to the missing image, which is _i_ = 2. _i_ = 1 refers to a previous time, and _i_ = 3 refers to a time after. L# and M#, as stated above, refers to the image provided at that time. For example, M2 is the "second" timestamp MODIS image, which is the one to be upscaled by the STTFN. L1 is the Landsat image in a previous timestamp.

The first formula is calculates the weight parameter for each CNN output. Essentially, the weight parameter for a specific CNN output is calculated by element-wise subtracting the given MODIS and the CNN output, estimating CNN accuracy. That accuracy is this used to show a preference towards a specific CNN output in that weight parameter.

The second formula, which then produces the final STTFN output prediction, uses the calculated weight parameters to find the most accurate estimation of the combined CNN outputs. 

# STARFM: What is it?

StarFM is an established baseline model we are comparing STTFN’s performance with in terms of  RMSE and SSIM. The model functions the same as the model listed in the original repo (https://github.com/nmileva/starfm4py), including use of the same parameters’ values.

For testing, we put the StarFM model inside a class to substantiate instances and make the model’s parameters part of its class.

# Using the .ipnyb

We encourage anyone looking to try out the .ipnyb file, though if you don't have access to University-level HPC Clusters, then the L4 GPU on the Google Colab should work just fine for this purpose.

Provide a named area of interest compatible with the Microsoft Planetary Computer connectors, and test your area of interest.

We have provided example areas of interest, named within the repository.

For the StarFM, you can change the TEST_POINT global variable in Imports to a different Test Data Point to test with. If TEST_POINT is >= the number of datapoints available, it defaults to the last data point available in trainingDataSet (check StarFM Test Run). Model outputs, including RMSE and SSIM are displayed in Testing Results.

Link to the Repository (if you are on pages): https://github.com/MahmoudFakhry/STTFN-Model
