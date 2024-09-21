# india-index-
Project 1
Hs code Prediction 
Creating an HS (Harmonized System) code prediction model using deep learning is an excellent application of machine learning in international trade. Here are the steps to build such a model:

Data Collection:

Gather a large dataset of product descriptions and their corresponding HS codes.
Sources include customs databases, international trade datasets, and government trade data.


Data Preprocessing:

Clean the text data (remove special characters, standardize case, etc.).
Tokenize the product descriptions.
Remove stop words and perform stemming or lemmatization.
Convert HS codes into appropriate target format (e.g., multi-class labels).


Feature Engineering:

Create word embeddings (e.g., Word2Vec, GloVe, or FastText).
USed  pre-trained embeddings specific to trade or general language models.


Data Split:

Divide the dataset into training, validation, and test sets.


Model Architecture:

Uses a deep learning model suitable for text classification. Options include:

 RNN (Recurrent Neural Network) with LSTM or GRU layers


 PRoject 2
 This code implements a deep learning model to predict economic impact based on various factors related to trade and economics. Here's a summary of its workflow:

1. Data Generation:
   - Created a synthetic dataset with 10,000 samples.
   - Features include tariff rates, trade volumes, GDP, exchange rates, political stability, and global demand.
   - Generates a complex, non-linear economic impact as the target variable.

2. Data Preprocessing:
   - Split the data into training and testing sets.
   - Scaled the features using StandardScaler.

3. Model Architecture:
   - Creates a Sequential neural network with Dense layers, BatchNormalization, and Dropout.
   - Used ReLU activation for hidden layers and a linear output layer.
   - Compiled the model with Adam optimizer and mean squared error loss.

4. Model Training:
   - Implemented early stopping to prevent overfitting.
   - Trains the model for up to 100 epochs with a batch size of 32.

5. Model Evaluation:
   - Calculates and printed the mean squared error for both training and test sets.

6. Visualization:
   - Ploted actual vs predicted economic impact values.
   - Displayed the training history (loss over epochs).

7. Feature Importance Analysis:
   - Used permutation importance to determine the most influential features.
   - Visualized feature importance with a bar plot.

8. Data Export and Prediction:
   - Saved the generated dataset to a CSV file.
   - Demonstrated how to use the trained model for prediction on new data.

This code provides a comprehensive approach to modeling complex economic relationships, including data generation, model training, evaluation, and interpretation of results.



