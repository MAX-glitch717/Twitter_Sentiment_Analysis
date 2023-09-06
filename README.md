# Twitter_Sentiment_Analysis

This project utilizes the power of Spark ML to predict sentiments (positive or negative) of live tweets. The project is implemented in a Jupyter notebook environment and was developed using Google Colab.

## Description

The primary goal of this project is to train a model using historical tweet data to classify the sentiments of incoming tweets into positive (label 0) or negative (label 1). The program leverages the capabilities of Spark's ML library to process the data, train a logistic regression model, and make predictions on live tweet data streamed from a local socket.

## Dependencies

- PySpark
- Spark ML libraries
- Spark Streaming

## Setup & Execution

1. Ensure you have Spark and PySpark set up in your environment.
2. Start a netcat server on port `9999` to simulate incoming tweets:
```bash
nc -lk 9999
```
3. Open the `twitter.ipynb` notebook in Jupyter or Google Colab.
4. Run the cells sequentially.
5. Input tweets into the netcat server to view the model's predictions in the notebook.

## Features

- Reading and preprocessing historical tweet data.
- Visualization and basic analysis of the training data.
- Tokenization and feature extraction of tweets.
- Training a logistic regression model using a Spark ML pipeline.
- Streaming live tweets from a local socket and making real-time predictions.

## Note

The current implementation predicts sentiments for individual words in a tweet. For more accurate results, consider processing and predicting sentiments for complete tweets.
