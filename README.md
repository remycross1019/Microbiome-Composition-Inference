# Code for Creating and Executing a Neural Network to Infer Microbiome Composition at Specific Threshold Values

### Process for Microbiome Inference
  1. Creating the Loss Function (LossFunctionMethods.py)
  2. Filtering the Data (Thresholds.py)
  3. Creating the Network Architecture (Architectures.py)
  4. Training and Scoring the Network (Network.py)


### Python Requirements
* numpy
* pandas
* skbio
* sklearn


### Scripts
#### 1. LossFunctionMethods.py
This script contains the loss function used to train the Neural Network as well as all of the helper methods used to create the loss function. It is based on SKBio's Weighted Unifrac Distance, however the methods have been modified so that they are compatible with Keras and tensors.

#### 2. Thresholds.py
This script tests the accuracy of the Neural Network in order to determine the most effective combination of threshold filters on both axes of the dataset. The data that is used is microbiome data taken from different tissue samples. On one axis is the sample ID for a tissue sample while on the other is the Taxonomic ID for the microbe. The entire dataset is fairly sparse, so adding in these thresholds changes the effectiveness of the NN. Thresholds are based on the abundance of samples and the prevalence of taxa.  

#### 3. Architectures.py
This script tests the accuracy of the Neural Network considering varying architectures and constant threshold values in order to determine the optimal architecture.

#### 4. Network.py
This script runs the Neural Network with user-inputted thresholds and architectures. Once the optimal thresholds and architecture are determined from the previous two scripts, this script allows us to specify both values to run the network.
