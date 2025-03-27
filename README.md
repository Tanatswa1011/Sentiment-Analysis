# Report for the Twitter Sentiment Analysis

## **Problem Statement:**

In today’s digital landscape, customer opinions and brand image is very important and is shaped hugely by discussions done in the online spaces like Twitter for example. Understanding the sentiments behind these discussions is very important for our company to assess customer satisfaction so that we can identify potential issues and minimize them before they become worse. This will also help them improve on our products and services.

This project aims to leverage sentiment analysis techniques to systematically evaluate Twitter comments. By analyzing these sentiments, we will be able to gain actionable insights to enhance our customer engagement, improve brand perception, and importantly make data-driven business decisions.

**Business Impact**

What business question does your model attempt to answer?

The goal of the model through NLP and Deep Learning is to:

- Automatically classify comments on Twitter as positive, neutral, and negative.
- Understand what topics/features are mentioned more in positive comments versus negative.

## System Architecture

High-Level Design

1. Data Cleaning: Tokenization & stopword removal.
2. Feature Extraction: Word embeddings via **Word2Vec**.
3. Model training: Training the **LSTM** models.
4. Evaluation and Insights: Model evaluation and insights generation.

Data Cleaning

## **Tokenization and Cleaning**

- Removed all punctuations and numbers.
- Converted all words to lowercase.
- Tokenization and stopwords removal were done using **NLTK**.
- We also performed lemmatization on the texts

## **Addressing Class Imbalance**

- The sentiment comments were classified into three sentiment classes:
    - **Positive**
    - **Neutral**
    - **Negative**
- Class imbalance was addressed through oversampling/under-sampling.

## Word2Vec for Feature Extraction

**Why Word2Vec?**

Word2Vec creates word embeddings purely based on relative association to other words. Word2Vec differs from TF-IDF in that it does not project meaning based on frequency of use; it simply knows based on the context of the use.

**Execution**

- Word2Vec was trained on the corpus to generate the word vectors.
- Pre-trained Word2Vec was also used to apply it to a larger database.
1. Sentiment Classification with LSTM

**Why LSTM?**

- LSTM is the recommended approach for sequential/time-series data.
- Tweets made on Twitter are time-series/sequentially dependent, and thus GRUs/LSTMs can learn from it.

**Architecture**

- **Embedding Layer**: Uses Word2Vec embedding.
- **Bidirectional LSTM Layer**: Reads the text and interprets it in both directions for more contextual relevance.
- **Dense Layer**: Fully connected layer for feature extraction.
- **Softmax Layer**: Provides output probabilities for each sentiment classification.

**Training and Optimization**

- Dropout rates were tuned to avoid overfitting.
- Adam optimizer and categorical cross-entropy were used to train the approach.
1. Results

## Performance Metrics

Classification Report:
                                 precision    recall       f1-score       support

```
Negative       0.78      0.78      0.78      2062
 Neutral       0.75      0.77      0.76      1958
Positive       0.79      0.77      0.78      1980

accuracy                           0.77      6000

```

macro avg            0.77      0.77      0.77      6000
weighted avg       0.77      0.77      0.77      6000

The classification report shows that the model performs relatively well for the sentiment classes—precision, recall, and F1-scores range from 0.77 to 0.79. As for accuracy and recall scores, the GRU performs marginally better than LSTM, meaning that for this task, a GRU would be more suitable, potentially due to its architecture being more effective in representing temporal dependencies within this type of data.

Thus, these metrics are essential for comprehending the strengths and weaknesses of the model.

## Libraries used and what for

**Data Manipulation and Analysis**

- pandas - to read data as a DataFrame
- numpy - to read numerical arrays/sets of information

**Natural Language Processing (NLP)**

- nltk - to tokenize, remove stopwords, and lemmatize.
- TextBlob - for sentiment analysis
- gensim - for topic modeling and Word2Vec use

**Machine Learning**

- scikit-learn - to train/test and evaluate as well as the preprocessing used.
- keras - to create and train the LSTM deep learning model.

**Visualization**

- matplotlib - to visualize with static and interactive options.
- seaborn - for statistical graphics generation.
- WordCloud - for visualizing word frequency.

**Miscellaneous**

- warnings - to suppress warnings in the environment.
- collections - alternative containers, including Counter.
    
    Key Functionality Accomplished
    
- **Data Cleaning**: Text cleaning, stop word removal, lemmatization.
- **Data Visualization**: Word clouds, sentiment distribution graphs.
- **Model Generation**: Bidirectional LSTM model generation and training.
- **Sentiment Assessment**: Assessment of sentiment for comments provided by user.
- **Testing Outputs**: Confusion matrices, classification reports.
1. Reflection

Summary

- Implemented **Word2Vec + LSTM** to achieve sentiment analysis.
- Classification of positive and negative sentiment of comments from Twitter achieved high accuracy.
- Provided useful findings based upon derived information.

Limitations and Future Research Considerations

- Model might not effectively generalize to future comments with new slang/expression.
- Future considerations should implement **transformer-based models (BERT)** for deeper context understanding.
- Future studies should include aspect-based sentiment analysis for more specific information about user complaints.
https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
