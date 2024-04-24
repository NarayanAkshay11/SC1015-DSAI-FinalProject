# SC1015-DSAI-FinalProject
### Fake News Detection using NLP, Machine Learning and Deep Learning.
## Problem Statement
Our project focuses on creating a comprehensive analysis framework to classify news articles as either genuine or fake. Leveraging features such as keywords, subject matter, and country of origin, we aim to build models capable of accurately determining the truthfulness index of news articles on a scale of 1 to 5. The goal is to provide a tool for combating misinformation and promoting the dissemination of accurate information in the digital landscape. We will be looking at data largely based in North America & India.

## Background
In recent years, the rapid exchange of information has underscored significant technological deficiencies that impact people's lives. The proliferation of social media has accelerated the spread of misinformation by malicious actors worldwide. Our group aims to address this challenge by leveraging Machine Learning principles acquired in our coursework to develop a solution. Our project emerged from discussions with friends and peers, heightening our awareness of the issue. Historical and contemporary instances demonstrate how misinformation can incite chaos and even cost lives. Ultimately, our group seeks to promote peace and stability by employing Machine Learning to distinguish between fake and genuine news.
Our motivation stems from the urgent need to address the rampant spread of misinformation in news articles. In today's digital age, fake news undermines public trust, distorts factual information, and leads to societal harm. By leveraging machine learning techniques, we aim to develop a solution that accurately identifies fake news, empowering individuals and organizations to make informed decisions and combat the spread of misinformation.

## The Team 😀 - This gives their LinkedIn
[Akshay Narayanan Balajee Viswanath](https://www.linkedin.com/in/akshay-narayanan-b-655a4023a/)

[Maanya Juneja](https://www.linkedin.com/in/maanya-juneja-1059542a3/)

[Aryan Garg](#)

## Data Collection
Initial Datasets:
https://huggingface.co/datasets/liar

https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset

https://www.kaggle.com/datasets/imbikramsaha/fake-real-news




[All Processed Datasets](https://www.kaggle.com/datasets/akshaynarayananb/sc1015dsai-final-fce2-team-1-23-24)

## Data Cleaning
Data cleaning starts with the dropping of unnecessary columns. We dropped nearly 17-18 features in a single dataframe since it contains json links, unnamed coluns and other data.
Removal of null values and to ensure only meaningful sentences are there in the dataframe all sentences with less than 10 words were removed. Duplicates were dropped and states were dropped to get countries. For dataframes which have only T/F or Real/Fake they were edited to fit them all on a 0-5 Truthfulness scale, 0 being False and 5 being True. This helps for the regressor model.

## EDA & Visualization
Pie Chart for TruthRatings Distribution:
The pie chart provides a quick visual breakdown of the dataset by truthfulness rating. Notably, the majority of the dataset comprises articles with a TruthRating of 0, indicating a high prevalence of fake news. This imbalance was deliberate, given that fake news typically features intense, emotion-driven language. A larger sample of such articles may help train our algorithm to better identify these patterns and improve its accuracy in detecting fake news.

Histogram for Article Counts by Country:
Histograms are used to group data points into defined ranges or bins, aiding in the visualization of trends, patterns, or outliers. In this histogram, we examine the distribution of articles by country. It's clear that the number of American articles far exceeds the number of Indian ones, indicating an uneven representation in the dataset.

Dual-Variable Histogram for TruthRatings by Country:
This histogram shows the distribution of news articles across various truth ratings for each country. The side-by-side comparison reveals a significant disparity: American articles are predominantly of TruthRating 0, whereas Indian articles are more evenly distributed, with a slight skew toward TruthRating 5. This observation suggests that the likelihood of encountering fake news is higher among American sources.

Heatmap for Country vs. Subject:
A heatmap uses color intensity to represent data values within a matrix, making it a useful tool for identifying patterns or correlations. This heatmap illustrates the relationship between countries and subjects, showing that US governmental articles are more prevalent than any other country-subject combination. This, coupled with the high proportion of low TruthRating articles in the US, led us to hypothesize that a significant portion of US governmental articles would be of low truthfulness. We further explore this in the following visualization.

Histogram for Average TruthRating by Subject:
To test our hypothesis, we plotted a histogram of the average TruthRating by subject. This histogram shows that the average TruthRating for government articles is around 1.0, supporting our hypothesis that most government-related articles have low TruthRatings.

Word Cloud for Frequent Words in Fake News:
A word cloud is a visual representation of text data, where words with higher frequency appear larger. This tool helps identify the most common words in a given subset of data. Our word cloud focused on TruthRatings 0, 1, and 2 to highlight commonly used terms in fake news. The prominent words include "Fact Check," "new," "say," "look," "people," and "found," indicating typical language patterns found in fake news articles.

## Machine Learning Model - Linear Regression
Linear regression is a statistical technique used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation. In simple linear regression, you have one independent variable.

Y = mx + b

In multiple linear regression, you can have multiple independent variables. Linear regression uses methods like Ordinary Least Squares (OLS) to minimize the sum of squared errors.

Common evaluation metrics for linear regression are R-squared, Mean Squared Error (MSE), and F1 score. An accuracy of 0.777 and an F1 score of 0.762 indicates a reasonable performance, where the F1 score accounts for both precision and recall, balancing the trade-off between false positives and false negatives.


## Sequential Neural Networks
Neural networks are machine learning models inspired by the human brain, consisting of interconnected layers of nodes (neurons). The simplest type of neural network is the Sequential model, where data flows from one layer to the next in a straight line.

LSTM (Long Short-Term Memory): LSTMs are a type of recurrent neural network designed to handle sequential data, like text or time series, by maintaining a memory of previous inputs. This allows them to capture patterns and dependencies over time.
Embedding Layer: This layer converts categorical data, like words, into dense vector representations, useful in natural language processing (NLP) to understand semantic relationships.
Dense Layer: A fully connected layer where each neuron is connected to every neuron in the previous layer, often used at the end of a neural network for output.
F1 Score
The F1 score is a measure of a model's accuracy in terms of both precision and recall. Precision indicates the proportion of true positives among predicted positives, while recall shows the proportion of true positives among actual positives. The F1 score combines these two metrics, offering a more balanced view of a model's performance, particularly in scenarios where class imbalance is a concern.


## Evaluation
In our case, the linear regression model has an accuracy of 0.777 and an F1 score of 0.762. The neural network, on the other hand, has an accuracy of 83.5. While both metrics are useful, they measure different aspects of performance. Accuracy is a straightforward measure of correct predictions, whereas the F1 score accounts for the balance between precision and recall.

Given the higher accuracy of the neural network (83.5), it might be considered the better model for our data. However, the choice between models also depends on other factors, such as computational cost, interpretability, and specific application requirements. If our focus is on general prediction accuracy, the neural network is likely more suitable. If interpretability or understanding relationships between variables is crucial, linear regression might be preferable.

## References
https://www.simplilearn.com/tutorials/deep-learning-tutorial/neural-network 
https://www.ibm.com/downloads/cas/GB8ZMQZ3 


