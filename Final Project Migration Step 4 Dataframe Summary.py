import logging
import platform
import os
from collections import Counter
import subprocess
import sys


import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm.auto import tqdm

from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DistilBertModel, DistilBertTokenizer)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def set_mkl_threading_layer():
    """
    Dynamically sets the MKL_THREADING_LAYER environment variable to avoid conflicts
    between Intel OpenMP and LLVM OpenMP libraries, based on the operating system.
    """
    # Detect operating system
    os_name = platform.system().lower()
    
    # Default to INTEL for Windows to use Intel's OpenMP
    mkl_threading_layer = "INTEL" if os_name == "windows" else "GNU"
   
    os.environ["MKL_THREADING_LAYER"] = mkl_threading_layer
    print(f"Operating system detected: {os_name.capitalize()}. Set MKL_THREADING_LAYER to {mkl_threading_layer}")


def install_and_import(package):

    try:
    
        module = __import__(package)
        return module
    except ImportError as e:
        print(f"{package} not found. Installing {package}...")
   
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
     
        try:
            module = __import__(package)
            return module
        except ImportError:
            raise ImportError(f"Failed to import or install {package}.") from e


def load_xlsx_as_dfs(folder_path):

    dfs = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
          
            df_key = os.path.splitext(file_name)[0]
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path, engine='openpyxl')

            dfs[df_key] = df
            
    return dfs


def extract_keywords(texts, top_n=5):
    
    stop_words = list(ENGLISH_STOP_WORDS) + ['like', 'said', 'says', 'actually', 'probably', 'maybe', 'perhaps', 'really', 'just', 'quite', 'simply', 'so', 'very']

    all_words = []

    for text in texts:
        words = [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words and len(word) > 2]
        all_words.extend(words)

    most_common = Counter(all_words).most_common(top_n)
    keywords = [word for word, _ in most_common]
    return keywords


def find_main_themes_in_texts_bert(text_data, n_topics=5):
    logging.info('Starting to generate embeddings with BERT.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name).to(device)

    embeddings = []
    for text in tqdm(text_data, desc="Processing Texts"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())

    embeddings = np.array(embeddings)

    adjusted_n_topics = min(n_topics, len(set(text_data)))

    kmeans = KMeans(n_clusters=adjusted_n_topics, random_state=0)
    kmeans.fit(embeddings)
    document_topics = kmeans.labels_

    cluster_keywords = {}
    for label in set(document_topics):
        indices = [i for i, topic in enumerate(document_topics) if topic == label]
        cluster_texts = text_data.iloc[indices].tolist()  
        keywords = extract_keywords(cluster_texts)
        cluster_keywords[f"Cluster {label + 1}"] = ', '.join(keywords)

    topics_df = pd.DataFrame.from_dict(cluster_keywords, orient='index', columns=['Keywords'])

    return topics_df




def generate_summary(text, df_name, model_name='t5-small', max_length=1024, min_length=25, output_dir='~/Desktop/Text Mining Class/News Summaries'):
    output_dir = os.path.expanduser(output_dir)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory {output_dir} is ready.")
    except Exception as e:
        logging.error(f"Error creating output directory: {e}")
        return

    try:
        logging.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        logging.error(f"Error loading model/tokenizer: {e}")
        return
    
    try:
        logging.info("Generating summary...")
        inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=max_length, truncation=True)
        summary_outputs = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        
        summary = tokenizer.decode(summary_outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return
    
    try:
        output_file_name = f"{df_name}_summary.txt"
        output_file_path = os.path.join(output_dir, output_file_name)
        with open(output_file_path, 'w') as file:
            file.write(summary)
        
        logging.info(f"Summary saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Error saving summary: {e}")

    return summary
 
def generate_summary_for_dfs(dfs_dict):

    for df_name, df in dfs_dict.items():
  
        combined_text = " ".join(df['content'].tolist())
        
        generate_summary(combined_text, df_name)
        
        
def find_top_5_words_df(df):
  
    all_words = []
    for keywords_str in df['Keywords']:
        keywords_list = keywords_str.split(', ')  
        all_words.extend(keywords_list)
        
    word_counts = Counter(all_words)
   
    top_5_words = word_counts.most_common(5)
    
    print("Top 5 most common words:", [word for word, count in top_5_words])


set_mkl_threading_layer()

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

openpyxl = install_and_import('openpyxl')

folder_path = "/Users/weisheit/Desktop/News_Articles/editorial_stance_dfs"

dfs_dict = load_xlsx_as_dfs(folder_path)

n_topics = 30

if 'MAGA' in dfs_dict:
    topics_bert_MAGA_df = find_main_themes_in_texts_bert(dfs_dict['MAGA']['content'], n_topics)

if 'Mainstream' in dfs_dict:
    topics_bert_Mainstream_df = find_main_themes_in_texts_bert(dfs_dict['Mainstream']['content'], n_topics)


generate_summary_for_dfs(dfs_dict)

top_5_words_mainstream = find_top_5_words_df(topics_bert_Mainstream_df)

top_5_words_maga = find_top_5_words_df(topics_bert_MAGA_df)