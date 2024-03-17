# Beyond Borders: Dissecting the Immigration Debate Through Sentiment Analysis

## Introduction

This project, conceived by Eric Brattin under the guidance of Dr. Gates for IST 736, dives into the complex and often polarized world of American immigration debate coverage in the mainstream media. Utilizing advanced machine learning (ML) and natural language processing (NLP) techniques, it aims to uncover whether the media's portrayal of immigration is skewed towards negativity and sensationalism or if it maintains a rational and informative stance.
By analyzing news articles from a wide range of media outlets, this project seeks to identify the prevailing narratives and sentiments surrounding immigration, offering insights into how these portrayals might influence public perception and policy discussions.

## Project Structure

- `Final Project Migration Step 1 Get Data.py` - Data collection script using NewsAPI and web scraping techniques.
- `Final Project Migration Step 2 Create Label with Vader and Bert.py` - Sentiment analysis using VADER and BERT models to label data.
- `Final Project Migration Step 3 Sklearn Predictions.py` - Machine learning predictions using the Multinomial Naive Bayes classifier.
- `Final Project Migration Step 4 Dataframe Summary.py` - Data analysis and visualization to summarize findings.

## Installation

To run this project, you will need Python 3.x and the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- beautifulsoup4
- requests
- nltk
- transformers

## Usage

1. **Data Collection:** Run `Final Project Migration Step 1 Get Data.py` to collect news articles on immigration.
2. **Data Labeling:** Execute `Final Project Migration Step 2 Create Label with Vader and Bert.py` to label the collected articles using sentiment analysis.
3. **Prediction:** Use `Final Project Migration Step 3 Sklearn Predictions.py` to apply the Multinomial Naive Bayes classifier for sentiment prediction.
4. **Analysis:** Analyze the results with `Final Project Migration Step 4 Dataframe Summary.py` for a comprehensive overview of the sentiments and themes in the media coverage.

## Methodology

The project's methodology encompasses a blend of data collection, preprocessing, sentiment analysis, and statistical comparison techniques. Key highlights include:
- **Data Collection:** Systematic collection of nearly 3000 articles using NewsAPI and web scraping.
- **Sentiment Analysis:** Employing VADER and BERT for nuanced sentiment labeling.
- **Machine Learning:** Using the Multinomial Naive Bayes Classifier for accurate sentiment prediction.
- **Data Analysis:** Advanced techniques, including Decision Tree Visualization and Latent Dirichlet Allocation (LDA), to uncover prevalent media themes related to immigration.

## Conclusion

This project provides an in-depth analysis of the sentiment and framing of immigration in media coverage. Through detailed data collection and advanced analytical methods, it offers valuable insights into the media's role in shaping public discourse on immigration.
For more details on the findings and implications of this study, please refer to the full project documentation.

![image](https://github.com/TruthQuest/NewsVoyager/assets/108246429/417c48d0-5c47-45d0-9f17-6ad8dd876dd9)
