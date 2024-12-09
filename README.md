# ROUGE-SciQFS: A ROUGE-based Method to Automatically Create Datasets for Scientific Query-Focused Summarization

This is the resource repository for the paper *ROUGE-SciQFS: A ROUGE-based Method to Automatically Create Datasets for Scientific Query-Focused Summarization* by Ramirez-Orta et al.

## Abstract

So far, the tasks of Query-Focused Extractive Summarization and Citation Prediction (QFS/CP) have lagged in development when compared to other areas of Scientific Natural Language Processing because of the lack of data. In this work, we propose a methodology to take advantage of existing collections of academic papers to obtain large-scale datasets for these tasks automatically. After applying it to the papers from our reading group, we introduce the first large-scale dataset for QFS/CP, composed of 8,695 examples, each composed of a query, the sentences of the full text from a paper and the relevance labels for each. After testing several classical and state-of-the-art text representation models and classifiers on this data, we found that these tasks are far from being solved, although they are relatively straightforward for humans. Surprisingly enough, we found that classical models outperformed modern pre-trained deep language models (sometimes by a large margin), showing the need for large datasets to fine-tune the latter. We share our code, data and models for further development of these areas.

## Contents

* The **notebooks** folder contains the .ipynb notebooks describing the methodology presented in the paper. They are numbered in the way they should be followed to reproduce the data cleaning process and to obtain the tables in the paper.
* The **scripts** folder contains the actual Python scripts used to implement the experiments of the paper: computation of embeddings, data augmentation process and training of the models can all be found here.
* Finally, the **library** folder contains reusable Python code to foster development of the experiments presented here: the Python package contains some utility functions and the model architectures required to train models from scratch.

## Installation

Tested on Python 3.8.10. 
After decompressing the zip file:
    
    cd malnis_data
    
To download the dataset

    pip install requests tqdm
    python download_data.py
    
To install the utilities and models

    pip install -r requirements
    pip install .
