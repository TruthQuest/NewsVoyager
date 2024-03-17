
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json  
import logging
import os
import re
import shutil 
import uuid
import webbrowser


import joblib  
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go  
import torch
from fuzzywuzzy import fuzz
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm.auto import tqdm


nltk.download('vader_lexicon')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def load_excel_file(file_path, file_name):
    
    full_path = os.path.join(file_path, file_name)

    try:
        df = pd.read_excel(full_path)
        print(f"Loaded {len(df)} rows from '{full_path}'")
        return df
    except FileNotFoundError:
        print(f"File not found: '{full_path}'")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    

def map_editorial_stance(data, media_dict, threshold=0.60):
 
    # Convert dictionary keys to a list for vectorization
    dict_keys = list(media_dict.keys())
    
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    
    # Fit the vectorizer on the combined set of publisher names
    vectorizer.fit(list(data['publisher']) + dict_keys)
    
    # Transform both dataset publisher names and dictionary keys
    data_vec = vectorizer.transform(data['publisher'])
    dict_keys_vec = vectorizer.transform(dict_keys)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(data_vec, dict_keys_vec)
    
    # Initialize an empty list to store editorial stances
    stances = []
    
    for i in range(len(data)):
        max_sim_index = cosine_sim[i].argmax()
        max_sim_value = cosine_sim[i][max_sim_index]
        
        # Check if the highest similarity score meets the threshold
        if max_sim_value >= threshold:
            stances.append(media_dict[dict_keys[max_sim_index]])
        else:
            # Use linear_kernel as a backup
            linear_sim = linear_kernel(data_vec[i], dict_keys_vec).flatten()
            max_linear_index = linear_sim.argmax()
            stances.append(media_dict[dict_keys[max_linear_index]])
    
    # Assign the calculated stances to the 'editorial_stance' column
    data['editorial_stance'] = stances
    
    return data


def resolve_duplicates(df):
    # Drop any column that has a digit in it
    df = df.drop(columns=[col for col in df.columns if re.search(r'\d', str(col))])
    
    unnamed_columns = [col for col in df.columns if 'unnamed' in col.lower()]
    for col in unnamed_columns:
        logging.info(f"Dropping column '{col}' as it contains 'unnamed' string")
    df = df.drop(columns=unnamed_columns)
    
    columns_to_drop = []

    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns[i + 1:]):  
      
            similarity = fuzz.ratio(col1, col2)
            if similarity >= 95:
                logging.info(f"Columns '{col1}' and '{col2}' are considered duplicates with {similarity}% similarity.")
     
                columns_to_drop.append(col2)
    
    df = df.drop(columns=set(columns_to_drop))
    
    new_column_names = {col: col[:-2] if col.endswith(('_x', '_y')) else col for col in df.columns}
    df.rename(columns=new_column_names, inplace=True)
    
    nonnull_counts = df.notnull().sum()
    sorted_cols = nonnull_counts.sort_values(ascending=False).index
    df = df[sorted_cols]
    
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    
    if df.index.duplicated().any():
        logging.warning('Duplicate indices found. Resolving...')
        df = df.reset_index(drop=True)
    
    return df



def add_unique_id(df):

    df['unique_id'] = df.apply(lambda _: uuid.uuid4(), axis=1)
    return df



def analyze_sentiment_vader(df, text_column):
    logger.info("Starting sentiment analysis with VADER")

    sia = SentimentIntensityAnalyzer()  

    emotion_tone_scores = []
    overall_emotion = []
    emotion_intensity = []

    for text in df[text_column]:
        scores = sia.polarity_scores(text)
        emotion_tone_scores.append(scores)
        overall_emotion.append('positive' if scores['compound'] > 0 else 'negative')
        emotion_intensity.append(scores['compound'])

    df['emotion_tone_scores_vader'] = emotion_tone_scores
    df['overall_emotion_vader'] = overall_emotion
    df['emotion_intensity_vader'] = emotion_intensity

    logger.info("Sentiment analysis with VADER completed")

    return df


def analyze_sentiment_bert(df, text_column, batch_size=64, max_length=256):
    logger.info("Starting sentiment analysis with BERT")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)

    encoded_input = tokenizer(df[text_column].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    dataset = TensorDataset(encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset))

    sentiment_scores = []

    # Use torch.no_grad() to reduce memory consumption
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = softmax(logits, dim=1).cpu().numpy()
            sentiment_scores_batch = probabilities[:,1] - probabilities[:,0]
            sentiment_scores.extend(sentiment_scores_batch)

    sentiment_category = ["positive" if score > 0 else "negative" for score in sentiment_scores]
    df['sentiment_score_bert'] = sentiment_scores
    df['sentiment_category_bert'] = sentiment_category

    logger.info("Sentiment analysis with BERT completed")
    return df



def run_concurrent_sentiment_analysis_and_merge(df, text_column):
    if not df.index.is_unique:
        df.reset_index(inplace=True, drop=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_analysis = {
            executor.submit(analyze_sentiment_vader, df.copy(), text_column): 'vader',
            executor.submit(analyze_sentiment_bert, df.copy(), text_column): 'bert'
        }

        for future in as_completed(future_to_analysis):
            analysis_type = future_to_analysis[future]
            try:
                result = future.result()
                if 'unique_id' not in result.columns:
                    print(f"Error: 'unique_id' column missing in {analysis_type} result.")
                    continue

                if analysis_type == 'vader':
                    df = pd.merge(df, result[['unique_id', 'emotion_tone_scores_vader', 'overall_emotion_vader', 'emotion_intensity_vader']], on='unique_id', how='left')
                elif analysis_type == 'bert': 
                    df = pd.merge(df, result[['unique_id', 'sentiment_score_bert', 'sentiment_category_bert']], on='unique_id', how='left', suffixes=('', '_bert'))
                print(f"{analysis_type} analysis completed and merged.")
            except Exception as exc:
                print(f'{analysis_type} generated an exception: {exc}')

    return df




def filter_mismatched_sentiments(df):
    initial_row_count = len(df)
    

    filtered_df = df[df['overall_emotion_vader'] == df['sentiment_category_bert']]
    
    final_row_count = len(filtered_df)
    rows_dropped = initial_row_count - final_row_count
    percent_dropped = (rows_dropped / initial_row_count) * 100
    
    print(f"Rows before filtering: {initial_row_count}")
    print(f"Rows after filtering: {final_row_count}")
    print(f"Rows dropped: {rows_dropped}")
    print(f"Percent of rows dropped: {percent_dropped:.2f}%")
    
    return filtered_df


def filter_rows_by_keywords(df, column, keywords):
   
    pattern = '|'.join([f"(?i){keyword}" for keyword in keywords])
    
    filtered_df = df[~df[column].str.contains(pattern, na=False)]
    
    rows_filtered = len(df) - len(filtered_df)
    print(f"Filtered {rows_filtered} rows containing keywords in '{column}'.")
    
    return filtered_df


def calculate_influence_scores(twitter_followers):
    # Normalize the Twitter following to create influence scores
    total_followers = sum(twitter_followers.values())
    influence_scores = {outlet: followers / total_followers for outlet, followers in twitter_followers.items()}
    return influence_scores

def measure_media_bias_and_impact(df, twitter_followers):
    # Calculate the influence scores
    influence_scores = calculate_influence_scores(twitter_followers)
    
    # Convert influence scores to a DataFrame for a merge
    df_influence_scores = pd.DataFrame(list(influence_scores.items()), columns=['publisher', 'Influence Score'])
    
    # Group by stance and sentiment category to count articles
    grouped = df.groupby(['publisher', 'sentiment_category_bert']).size().reset_index(name='count')

    # Calculate total counts by stance
    total_counts = grouped.groupby('publisher')['count'].transform('sum')

    # Calculate traditional percentages within each stance
    grouped['traditional_percentage'] = (grouped['count'] / total_counts) * 100

    # Merge the influence scores into the grouped DataFrame
    grouped = grouped.merge(df_influence_scores, on='publisher', how='left')

    # Apply the influence score to the traditional percentage
    grouped['weighted_percentage_with_influence'] = grouped['traditional_percentage'] * grouped['Influence Score']

    # Pivot the data for a comprehensive view
    pivot_df = grouped.pivot_table(
        index='publisher', 
        columns='sentiment_category_bert', 
        values=['traditional_percentage', 'weighted_percentage_with_influence']
    ).fillna(0)
    
    # Rename the columns for clarity
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

    return pivot_df




media_outlet_stances = {
    "Aljazeera": "Mainstream",
    "BBC": "Mainstream",
    "Bloomberg": "Mainstream",
    "Breitbart": "MAGA",
    "Cnn": "Mainstream",
    "Chicago Tribune": "Mainstream",
    "Financial Times": "Mainstream",
    "Foxnews": "MAGA",
    "Los Angeles Times": "Mainstream",
    "NPR": "Mainstream",
    "Politico": "Mainstream",
    "Reuters": "Agnostic",
    "The American Conservative": "Conservative",
    "The Atlantic": "Progressive",
    "The Boston Globe": "Mainstream",
    "The Daily Wire": "MAGA",
    "The Economist": "Mainstream",
    "The Federalist": "MAGA",
    "The Guardian": "Progressive",
    "The National Review": "Conservative",
    "The New York Post": "MAGA",
    "The New York Times": "Mainstream",
    "Nymag": "Mainstream",
    "The Wall Street Journal": "Mainstream",
    "The Washington Examiner": "MAGA",
    "TWashingtonpost": "Mainstream",
    "Time": "Mainstream",
    "The Huffington Post": "Progressive",
    "Mother Jones": "Progressive",
    "The Intercept": "Progressive",
    "Jacobin": "Progressive",
    "Democracy Now!": "Progressive",
    "Salon": "Progressive",
    "Slate": "Progressive",
    "ThinkProgress": "Progressive",
    "The Nation": "Progressive",
    "Vox": "Progressive"
}

twitter_followers = {
    'Bbc': 2.3,
    'Aljazeera': 8.8,
    'Time': 19.3,
    'Politico': 4.6,
    'Nymag': 1.7,
    'Cnn': 62.2,
    'Washingtonpost': 20.0,
    'Breitbart': 2.0,
    'Foxnews': 24.4
}


file_path = os.path.expanduser('~/Desktop/News_Articles')

file_name = 'combined_df_web_content_deduplicated.xlsx'

combined_df_web_content_deduplicated = load_excel_file(file_path, file_name)

combined_df_web_content_deduplicated = add_unique_id(combined_df_web_content_deduplicated)

combined_vader_bert_df = run_concurrent_sentiment_analysis_and_merge(combined_df_web_content_deduplicated, 'content')

combined_vader_bert_df_deduped = resolve_duplicates(combined_vader_bert_df)

combined_vader_bert_df_filtered = filter_mismatched_sentiments(combined_vader_bert_df_deduped)

combined_vader_bert_df_filtered_mapped = map_editorial_stance(combined_vader_bert_df_filtered, media_outlet_stances)

editorial_sentiment_summary = measure_media_bias_and_impact(combined_vader_bert_df_filtered_mapped, twitter_followers)

combined_path = os.path.join(file_path, 'combined_vader_bert_df.xlsx')

combined_vader_bert_df_filtered_mapped.to_excel(combined_path, index=False)