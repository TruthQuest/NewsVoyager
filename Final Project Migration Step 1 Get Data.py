import json
import logging
import os
import re
import shutil
import subprocess
import sys
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlparse

from fuzzywuzzy import fuzz
import joblib
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm.auto import tqdm

nltk.download('vader_lexicon')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_and_upgrade(package, version=None):
    try:   
        module = __import__(package)     
        if version:
            import pkg_resources
            current_version = pkg_resources.get_distribution(package).version
            if pkg_resources.parse_version(current_version) < pkg_resources.parse_version(version):
                print(f"{package} version {current_version} is installed, but version {version} or newer is required. Upgrading...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
        
                module = __import__(package)
        return module
    except ImportError as e:
    
        print(f"{package} not found. Installing {package}...")
        install_command = [sys.executable, "-m", "pip", "install", package]
        if version:
            install_command[-1] += f"=={version}"
        
        subprocess.check_call(install_command)      
        try:

            module = __import__(package)
            return module
        except ImportError:
            raise ImportError(f"Failed to import or install {package}.") from e


def create_news_article_folders(desktop_path, query):
    """
    Note: 'AND' is used in 'download_news_stories_from_api' (below) to connect news themes/topics. 
    However, it's important that the file path does not include 'AND' so naming conventions are uniform.
    I'm removing AND with regex
    I want the file output to be example:  "news_articles_Ecuador_biodiversity" 
    where 2 topics are merged with underscore _
    
    """
    main_news_folder_name = "News_Articles"
    main_news_folder_path = os.path.join(desktop_path, main_news_folder_name)
    if not os.path.exists(main_news_folder_path):
        os.makedirs(main_news_folder_path)
        logging.info(f"Main folder '{main_news_folder_name}' now exists at {main_news_folder_path}")
    

    clean_query = re.sub(' +', ' ', query.replace('AND', ' ')).strip() 
    
    clean_query = re.sub(' ', '_', clean_query)  
    
 
    sub_folder_name = "news_articles_" + clean_query
  
    sub_folder_path = os.path.join(main_news_folder_path, sub_folder_name)
    if not os.path.exists(sub_folder_path):
        os.makedirs(sub_folder_path)
        logging.info(f"Folder '{sub_folder_name}' created here {sub_folder_path}")
        
    file_name = f"{clean_query}.json"
    file_path = os.path.join(sub_folder_path, file_name)
    
    return file_path


def load_access_key(file_path):
    """
    Note: This loads the API access key I saved in a txt file.
    
    It reads the full file content, removing any extra white spaces or blanks     
    """
   
    full_file_path = os.path.expanduser(file_path)
   
    with open(full_file_path, 'r') as file:
        return file.read().strip()
    

def find_matching_news_sources(api_key, list_of_news_outlets, similarity_score_threshold=0.50):
    
    """
    Note: This API has unique string names for news outlets. Example: the-new-york-times.
    Because of this, I am trying to get a list of how each media outlet appears as a string
    I want to pass a list to find  New York Times, even if its listed as new-york-times, 
    1st, I need to find all news outlets that exist in this API
    2nd, Vectorize API version of news outlet names and news outlets provided as input: "list_of_news_outlets"
    3rd, Use cosine similarity scores to compare and connect news outlet names
    4th, if similarity score is aove 70% news outlets are matched and a list of all matched news source is used to then filter 

    """
    
    news_api_client = NewsApiClient(api_key=api_key)
       
    all_sources_response = news_api_client.get_sources()
    
    all_available_news_outlets = all_sources_response.get('sources', [])
    
    available_news_outlet_names = [source['name'] for source in all_available_news_outlets]
        
    text_analyzer = TfidfVectorizer().fit(available_news_outlet_names + list_of_news_outlets)
    
    available_source_text_vectors = text_analyzer.transform(available_news_outlet_names)
    
    search_outlet_text_vectors = text_analyzer.transform(list_of_news_outlets)
    
    matched_source_ids = []
    
    for outlet_vector in search_outlet_text_vectors:
        
        similarity_scores = cosine_similarity(outlet_vector, available_source_text_vectors)
        
        highest_similarity_index = similarity_scores.argmax()
        
        if similarity_scores[0, highest_similarity_index] >= similarity_score_threshold:
            
            matched_source_ids.append(all_available_news_outlets[highest_similarity_index]['id'])
            
    return matched_source_ids



def download_news_stories_from_api(api_key, topics, source_identifiers=None, article_publish_date_start=None, article_publish_date_end=None):
    newsapi = NewsApiClient(api_key=api_key)
    desktop_path = os.path.expanduser('~/Desktop')
    all_stories = []

    for topic in topics:
        try:
            # If the topic is a list of strings, join with " AND " to form a SQL-like AND. If not, use the topic as is
            query = " AND ".join(topic) if isinstance(topic, list) else topic

            from_date = datetime.strptime(article_publish_date_start, '%Y-%m-%d') if article_publish_date_start else None
            to_date = datetime.strptime(article_publish_date_end, '%Y-%m-%d') if article_publish_date_end else None
            
            # Request artciles based query, source, and date
            
            response = newsapi.get_everything(q=query, sources=','.join(source_identifiers) if source_identifiers else None, from_param=from_date, to=to_date)
            
            articles = response['articles']
            for article in articles:
                # connect query to the article so we can use it as the label
                article['search_keyword'] = query
                all_stories.append(article)
                
            logging.info(f"Found {len(articles)} articles for topic '{query}'")
            
            file_path = create_news_article_folders(desktop_path, query)
            with open(file_path, 'w') as file:
                json.dump(articles, file)
                
            logging.info(f"News story with theme of '{query}' saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to process topic '{query}': {e}")

    return all_stories


def get_news_site_name(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
   # Extract the publisher's name from the domain, as its right after www.
    if domain.startswith('www.'):
        domain = domain[4:]
    publisher = domain.split('.')[0]
    # Replace hyphens with spaces and capitalize each word tp fix formatting
    publisher = ' '.join([word.capitalize() for word in publisher.split('-')])
    return publisher


def scrape_and_save_content_from_url_concurrently(df, query):
    if df.empty:
        print("The DF is empty. No URLs to scrape. Check that it has content")
        return pd.DataFrame()

    def scrape_content_from_url(url):
        print(f"Scraping URL: {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"You retrieved content from {url}")
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title').get_text()
                author = None
                author_tag = soup.find('meta', attrs={'name': 'author'})
                if author_tag:
                    author = author_tag['content']
                else:
                    # Search for an author in text
                    for tag in soup.find_all(['p', 'div'], text=True):
                        text = tag.text.strip()
                        if text.lower().startswith('by '):
                            author_name = text[3:]
                            words = author_name.split()
                            if len(words) <= 3:
                                author = ' '.join(words)
                                break
                date_tag = soup.find('meta', attrs={'name': 'publish-date'})
                publish_date = date_tag['content'] if date_tag else None
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text() for p in paragraphs])
                publisher = get_news_site_name(url)
                return {
                    'title': title,
                    'author': author,
                    'publish_date': publish_date,
                    'content': article_text,
                    'theme': query,
                    'url': url,
                    'publisher': publisher
                }
            else:
                logging.warning(f"Failed to pull content from {url}")
        except Exception as e:
            logging.error(f"Error scraping URL {url}: {e}")
            return None

    web_content_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(scrape_content_from_url, url): url for url in df['url']}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if result:
                    web_content_data.append(result)
            except Exception as exc:
                print(f'URL {url} generated an exception: {exc}')
    
    return pd.DataFrame(web_content_data)


def drop_empty_cols_fill_na(df):
    
    cols_to_drop = []
    for col in df.columns:

        if df[col].isna().all():
            cols_to_drop.append(col)
 
        elif (df[col] == 0).all():
            cols_to_drop.append(col)

    df.drop(columns=cols_to_drop, inplace=True)

    df.fillna("", inplace=True)
    
    return df


def remove_unwanted_words(df, col_name):
    
    # using ^ for "everything but", matches any character that is not a letter, number or whitespace
    df[col_name] = df[col_name].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    
    # remove words that contain numbers or dont meet length requirements
    df[col_name] = df[col_name].apply(lambda x: ' '.join([word for word in x.split() if word.isalpha() and 3 <= len(word) <= 13]))

    return df


def filter_short_text_rows(df):

    def has_minimal_content(content):
        if pd.isna(content) or len(str(content).strip()) < 1:
            return True
        return False

    df = df[~df['content'].apply(has_minimal_content)]

    return df


def save_dfs_to_excel(folder_path):

    news_folder_path = os.path.join(folder_path, 'News_Articles')
    if not os.path.exists(news_folder_path):
        os.makedirs(news_folder_path)
        print(f"Folder '{news_folder_path}' created.")

    for var_name, df in globals().items():
        if isinstance(df, pd.DataFrame) and var_name.startswith('df_web_content_combined'):
            file_path = os.path.join(news_folder_path, f"{var_name}.xlsx")
            df.to_excel(file_path, index=False)
            print(f"Saved {var_name} to {file_path}")
            
            
def move_files_to_correct_folders(source_destination_path):

    for file_name in os.listdir(source_destination_path):
        if file_name.endswith('.xlsx'):
    
            parts = file_name.split('_')
            if len(parts) < 4:  
                print(f"Skipping file due to naming convention: {file_name}")
                continue

            topic_keywords = parts[1:-1]  
            folder_pattern = '_'.join(topic_keywords).capitalize()  

            destination_folder = None
            for folder_name in os.listdir(source_destination_path):
                folder_path = os.path.join(source_destination_path, folder_name)
                if os.path.isdir(folder_path) and all(keyword.lower() in folder_name.lower() for keyword in topic_keywords):
                    destination_folder = folder_path
                    break
            
            if destination_folder:
           
                source_file_path = os.path.join(source_destination_path, file_name)
                destination_file_path = os.path.join(destination_folder, file_name)
                shutil.move(source_file_path, destination_file_path)
                print(f"Moved {file_name} to {destination_folder}")
            else:
                print(f"No matching folder found for {file_name}")
                
                
def find_combine_and_deduplicate_excel_files(desktop_path, filename="df_web_content_combined.xlsx"):
  
    news_articles_path = os.path.join(desktop_path, 'News_Articles')

    dfs = []
    
    for root, dirs, files in os.walk(news_articles_path):
        for file in files:
   
            if file == filename:
          
                full_path = os.path.join(root, file)
         
                df = pd.read_excel(full_path)
           
                dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)

    rows_before = combined_df.shape[0]

    combined_df = combined_df.drop_duplicates(keep='first')

    rows_after = combined_df.shape[0]
    
    rows_removed = rows_before - rows_after
    
    print(f"Rows removed: {rows_removed}")
    return combined_df


def filter_rows_by_keywords(df, column, keywords):
   
    pattern = '|'.join([f"(?i){keyword}" for keyword in keywords])
    
    filtered_df = df[~df[column].str.contains(pattern, na=False)]
    
    rows_filtered = len(df) - len(filtered_df)
    print(f"Filtered {rows_filtered} rows containing keywords in '{column}'.")
    
    return filtered_df


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


def extract_publisher(title):

    if ' - ' in title:
        return title.split(' - ')[-1]
    elif ' | ' in title:
        return title.split(' | ')[-1]
    else:
        return 'Unknown'  
    
    

install_and_upgrade('openpyxl', '3.1.0')


source_names = [
    "Al Jazeera English",
    "BBC News",
    "Bloomberg",
    "Breitbart News",
    "CNN",
    "Chicago Tribune",
    "Financial Times",
    "Fox News",
    "Los Angeles Times",
    "NPR",
    "Politico",
    "Reuters",
    "The American Conservative",
    "The Atlantic",
    "The Boston Globe",
    "The Daily Wire",
    "The Economist",
    "The Federalist",
    "The Guardian",
    "The National Review",
    "The New York Post",
    "The New York Times",
    "The New Yorker",
    "The Wall Street Journal",
    "The Washington Examiner",
    "The Washington Post",
    "Time",
    "The Huffington Post",
    "Mother Jones",
    "The Intercept",
    "Jacobin",
    "Democracy Now!",
    "Salon",
    "Slate",
    "ThinkProgress",
    "The Nation",
    "Vox"
]





simple_topics_for_api = [
    "refugees",
    "immigration AND policy",
    "border security",
    "asylum seekers",
    "migrant workers",
    "immigrant rights",
    "deportation AND policy",
    "undocumented immigrants",
    "refugee camps",
    "human trafficking",
    "immigration reform",
    "visa policy",
    "sanctuary cities",
    "integration AND immigrants",
    "cross-border movement",
    "climate refugees",
    "family reunification",
    "labor market integration",
    "anti-immigration policies",
    "refugee education",
    "migration and health",
    "border technology",
    "international asylum laws",
    "immigrant detention centers",
    "cultural integration"
]

desktop_path = os.path.expanduser('~/Desktop')

news_articles_path = os.path.join(desktop_path, 'News_Articles')

news_article_folders = [
    "news_articles_Refugees_Crisis/Refugees_Crisis.json",
    "news_articles_Immigration_Policy/Immigration_Policy.json",
    "news_articles_Border_Security/Border_Security.json",
    "news_articles_Asylum_Seekers/Asylum_Seekers.json",
    "news_articles_Migrant_Workers/Migrant_Workers.json",
    "news_articles_Immigrant_Rights/Immigrant_Rights.json",
    "news_articles_Deportation_Policy/Deportation_Policy.json",
    "news_articles_Undocumented_Immigrants/Undocumented_Immigrants.json",
    "news_articles_Refugee_Camps/Refugee_Camps.json",
    "news_articles_Human_Trafficking/Human_Trafficking.json",
    "news_articles_Immigration_Reform/Immigration_Reform.json",
    "news_articles_Visa_Policy/Visa_Policy.json",
    "news_articles_Sanctuary_Cities/Sanctuary_Cities.json",
    "news_articles_Integration_Immigrants/Integration_Immigrants.json",
    "news_articles_Cross_Border_Movement/Cross_Border_Movement.json",
    "news_articles_Climate_Refugees/Climate_Refugees.json",
    "news_articles_Family_Reunification/Family_Reunification.json",
    "news_articles_Labor_Market_Integration/Labor_Market_Integration.json",
    "news_articles_Anti_Immigration_Policies/Anti_Immigration_Policies.json",
    "news_articles_Refugee_Education/Refugee_Education.json",
    "news_articles_Migration_And_Health/Migration_And_Health.json",
    "news_articles_Border_Technology/Border_Technology.json",
    "news_articles_International_Asylum_Laws/International_Asylum_Laws.json",
    "news_articles_Immigrant_Detention_Centers/Immigrant_Detention_Centers.json",
    "news_articles_Cultural_Integration/Cultural_Integration.json"
]

api_key_file_path = os.path.join(desktop_path, "news_api_key.txt")

news_articles_path_full = [os.path.join(news_articles_path, path) for path in news_article_folders]

api_key = load_access_key(api_key_file_path)

source_identifiers = find_matching_news_sources(api_key, source_names)

all_stories = download_news_stories_from_api(api_key, simple_topics_for_api, source_identifiers=source_identifiers)

df_all_stories = pd.DataFrame(all_stories)

unique_queries = df_all_stories['search_keyword'].unique()

df_web_content_combined = pd.DataFrame()

for query in unique_queries:
    
    df_subset = df_all_stories[df_all_stories['search_keyword'] == query]

    df_web_content = scrape_and_save_content_from_url_concurrently(df_subset, query)
    df_web_content_combined = pd.concat([df_web_content_combined, df_web_content], ignore_index=True)
    
desktop_path = os.path.expanduser('~/Desktop')

df_web_content_combined_all = find_combine_and_deduplicate_excel_files(desktop_path)

df_web_content_combined_all_cleaned_1 = remove_unwanted_words(df_web_content_combined_all, 'content')

df_web_content_combined_all_cleaned_2 = filter_short_text_rows(df_web_content_combined_all_cleaned_1)

df_web_content_combined_all_cleaned_3 = resolve_duplicates(df_web_content_combined_all_cleaned_2)

#removing crucial refugee populations in history that are not currently relevant to the US Immigration debate
keywords = ["Israel", "Israeli", "Netanyahu", "Gaza", "Hamas", "UNRWA", 
            "Pakistan", "Rafah", 'Rohingya', 'Burma', 'Myanmar', 'Syria', "Jew",
            "Holocaust"]

df_web_content_combined_all_cleaned_4 = filter_rows_by_keywords(df=df_web_content_combined_all_cleaned_3, column='title', keywords=keywords)

news_articles_path = os.path.join(desktop_path, 'News_Articles')

combined_path = os.path.join(news_articles_path, 'combined_df_web_content_deduplicated.xlsx')

df_web_content_combined_all_cleaned_4.to_excel(combined_path, index=False)

