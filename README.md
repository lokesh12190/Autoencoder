# Autoencoder
Deep Autoencoders model for Anomaly Detection

pandas==1.3.4
keras==2.7.0
Flask==2.0.2
gunicorn
numpy==1.19.5
tensorflow==2.7.0

"Objective:
This project focuses on the application of Autoencoders in Deep Learning, particularly for learning compressed representations of data. Autoencoders consist of two main components: an Encoder and a Decoder. This project aims to provide a comprehensive understanding of Autoencoders and their architectural design. The primary objective is to develop a deep learning model using Autoencoders, specifically tailored for Anomaly Detection. Furthermore, the model will be deployed using Flask, a popular web framework. This project is part of a larger series dedicated to exploring various aspects of Deep Learning. For those interested in previous projects in this series, further information is available."

Aim :
● To understand the theory behind Autoencoders
● To develop a deep learning model based on Autoencoders to learn distributions
and relationships between features of normal transactions
● To deploy the model using Flask

Data Overview:
The dataset used is a transaction dataset, and it contains information for more than
100K transactions over several features.
Tech Stack:
➔ Language: Python
➔ Packages: Pandas, Numpy, matplotlib, Keras, Tensorflow
➔ API service: Flask, Gunicorn
Approach
1. Understand business objective
2. Understand the data using EDA
3. Normalize and clean the data using Imputation
4. Understand idea behind Auto-encoders
5. Build a base auto-encoder model using Keras
6. Tune the model to extract the best performance
7. Extract predictions
8. Serve model as API endpoint using Flask
9. Perform real-time predictions



1. The input folder contains all the data that we have for analysis. In our case, it will
contains two csv files which are
a. final_cred_data.csv
b. Test-data.csv
2. The src folder is the heart of the project. This folder contains all the modularized
code for all the above steps in a modularized manner. It further contains the
following.
a. ML_pipeline
b. engine.py
The ML_pipeline is a folder that contains all the functions put into different python
files which are appropriately named. These python functions are then called inside the
engine.py file
3. The output folder contains all the models that we trained for this data saved as
.pkl files. These models can be easily loaded and used for future use and the
user need not have to train all the models from the beginning.
4. The lib folder is a reference folder. It contains the original ipython notebook that
we saw in the videos.
5. The requirements.txt file has all the required libraries with respective versions.
Kindly install the file by using the command pip install -r requirements.txt
