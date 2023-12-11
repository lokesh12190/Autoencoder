# Autoencoder
Deep Autoencoders model for Anomaly Detection

"Objective:
This project focuses on the application of Autoencoders in Deep Learning, particularly for learning compressed representations of data. Autoencoders consist of two main components: an Encoder and a Decoder. This project aims to provide a comprehensive understanding of Autoencoders and their architectural design. The primary objective is to develop a deep learning model using Autoencoders, specifically tailored for Anomaly Detection. Furthermore, the model will be deployed using Flask, a popular web framework. This project is part of a larger series dedicated to exploring various aspects of Deep Learning."

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


## Dataset
The dataset contains features from credit card transactions, including:
- Timestamp: Time of the transaction.
- Value: Transaction amount.
- C1 to C12: Various anonymized features related to each transaction.
- Class: Target variable indicating whether the transaction is fraudulent (1) or not (0).

## Technical Approach
1. **Data Preprocessing**:
   - Data is loaded using Pandas and preprocessed to ensure it's in a suitable format for analysis.
   - Null values are handled, and data normalization is performed.
   - Exploratory Data Analysis (EDA) includes kernel density estimation to understand feature distributions.

2. **Model Development**:
   - A deep autoencoder model is built with TensorFlow and Keras.
   - The model comprises dense layers with ReLU activation for the encoder and linear activation for the decoder.

3. **Model Training and Evaluation**:
   - The model is trained using the preprocessed data, focusing on reconstructing the input features.
   - Training involves 20 epochs with a batch size of 128.
   - Model performance is evaluated using mean squared error (MSE) as the primary metric.

4. **Anomaly Detection**:
   - Post-training, the model's ability to reconstruct transactions is used to identify anomalies.
   - Transactions with high reconstruction errors (higher MSE) are flagged as potential frauds.

## Technologies Used
- Python: Primary programming language.
- Scikit-learn: Used for data processing and additional modeling tasks.
- TensorFlow and Keras: For building and training the deep learning model.
- Pandas: Data manipulation and analysis.
- Numpy: Numerical operations.

## Model Evaluation and Decision Making
- The model’s effectiveness is assessed based on its mean squared error (MSE).
- The best-performing model, in terms of the lowest MSE, is chosen for deployment.

## Deployment
- The final model is deployed for real-time risk assessment of transactions.
- The deployment is designed to be scalable and capable of making quick assessments.

## Repository Structure
- `Deep-Autoencoder.ipynb`: Contains the detailed process of model building, training, and evaluation.
- `Model_Api.ipynb`: Demonstrates the API implementation for deploying the model in a real-world application.

1. The input folder contains all the data that we have for analysis. In our case, it willcontains two csv files which are
a. final_cred_data.csv
b. Test-data.csv
2. The src folder is the heart of the project. This folder contains all the modularizedcode for all the above steps in a modularized manner. It further contains the
following.
a. ML_pipeline
b. engine.py
The ML_pipeline is a folder that contains all the functions put into different pythonfiles which are appropriately named. These python functions are then called inside the
engine.py file
3. The output folder contains all the models that we trained for this data saved as.pkl files. These models can be easily loaded and used for future use and the
user need not have to train all the models from the beginning.
4. The lib folder is a reference folder. It contains the original ipython notebook thatwe saw in the videos.
5. The requirements.txt file has all the required libraries with respective versions.install the file by using the command pip install -r requirements.txt

