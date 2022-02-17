# Data

Here is where we are keeping all the datasets we will be using. The original dataset is the [CICDarknet2020](https://www.unb.ca/cic/datasets/darknet2020.html) dataset.

The datasets are not kept in the github repository because they are too large to be hosted on github.
Anyone interested in running the experiments hosted on this repo need to follow the following steps:
 * Download the dataset from the [CICDarknet2020 website.](https://www.unb.ca/cic/datasets/darknet2020.html)
 * Place the darknet.csv file into the original directory contained within this one.
 * Install the required packages
 * Open Dataset_Cleaning.ipynb
 * click Run All to generate the basic cleaned dataset



This repository contains the following notebooks:
 * Dataset_Cleaning.ipynb: This notebook contains the code to clean the dataset before feature selection
 * Dataset_statistics.ipynb: This notebook contains the code used to explore the datasets and is used to collect any statistics about the datasets.
 * Feature_Selection.ipynb: This notebook is used to prune features with justification for removal.