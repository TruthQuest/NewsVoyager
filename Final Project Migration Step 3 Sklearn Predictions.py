import json
import logging
import os
import re
import shutil
import webbrowser
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlparse

import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel
from sklearn.cluster import KMeans
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm.auto import tqdm


nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')


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



def multinomial_nb_trainer(X, y, use_grid_search=True):
    
    custom_stop_words = list(ENGLISH_STOP_WORDS) + ['like', 'said', 'says']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=custom_stop_words, token_pattern=r'(?u)\b[a-zA-Z]{3,13}\b')),
        ('clf', MultinomialNB())
    ])
    
    if use_grid_search:
        parameters = {
            'tfidf__max_df': (0.5, 0.75, 1.0),
            'tfidf__ngram_range': ((1, 1), (1, 2)),
            'clf__alpha': (1e-2, 1e-3, 1e-4)
        }
        grid_search = GridSearchCV(pipeline, parameters, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        
        performance_report = {
            "Best Model's Parameters": grid_search.best_params_,
            "Classification Report": classification_report(y_test, predictions, output_dict=True),
            "Confusion Matrix": confusion_matrix(y_test, predictions).tolist()  
        }
        
        joblib.dump(best_model, 'best_multinomial_nb_model.joblib')
        
        return best_model, performance_report

    else:
        multinomial_nb_model = pipeline.fit(X_train, y_train)
        predictions = multinomial_nb_model.predict(X_test)
        
        performance_report = {
            "Classification Report": classification_report(y_test, predictions, output_dict=True),
            "Confusion Matrix": confusion_matrix(y_test, predictions).tolist()  
        }
        
        joblib.dump(multinomial_nb_model, 'multinomial_nb_model.joblib')
        
        return multinomial_nb_model, performance_report



def analyze_sentiment_with_multinomial_nb_trainer(df, text_column, model):
   
    df['predicted_sentiment_multinomialnb'] = model.predict(df[text_column])
    
    return df


def perform_lda_topic_modeling(df, text_column, n_topics=5, n_words=10):
    
    custom_stop_words = list(ENGLISH_STOP_WORDS) + ['like', 'said', 'says']

    count_vectorizer = CountVectorizer(stop_words=custom_stop_words,
                                       token_pattern=r'(?u)\b[a-zA-Z]{3,13}\b')

    dt_matrix = count_vectorizer.fit_transform(df[text_column].values.astype('U'))

    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(dt_matrix)
    
    words = count_vectorizer.get_feature_names_out()

    topic_words = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [words[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topic_words[f"Topic {topic_idx}"] = top_words

    topics_df = pd.DataFrame(topic_words)

    return lda_model, dt_matrix, topics_df


def assign_lda_labels(df, dt_matrix, topics_df, label_prefix="Topic"):

    dominant_topic_indices = np.argmax(dt_matrix.toarray(), axis=1)  
    descriptive_labels = topics_df.apply(lambda x: ', '.join(x.dropna().values.tolist()), axis=0)
    descriptive_labels = descriptive_labels.to_dict() 

    df['LDA_label'] = [descriptive_labels[f"{label_prefix} {i}"] for i in dominant_topic_indices]

    return df




def train_and_visualize_decision_trees(X_train, y_train, criterion='gini', max_depths=[None, 10, 5]):
    dashboard_links = [] 
    tree_metrics = []

    for max_depth in max_depths:
        dtree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
        dtree.fit(X_train, y_train)

        n_nodes = dtree.tree_.node_count
        children_left = dtree.tree_.children_left
        children_right = dtree.tree_.children_right
        feature = dtree.tree_.feature
        threshold = dtree.tree_.threshold

        node_x, node_y, hover_text = [], [], []

        def traverse(node, depth):
            if children_left[node] != children_right[node]:
                y = depth
                x = (traverse(children_left[node], depth + 1) + traverse(children_right[node], depth + 1)) / 2
                node_x.append(x)
                node_y.append(y)
                hover_text.append(f"Split on feature {feature[node]} at <= {threshold[node]:.2f}, Depth: {depth}")
            else:
                x = len([yx for yx in node_y if yx == depth])
                node_x.append(x)
                node_y.append(depth)
                hover_text.append(f"Leaf node, Depth: {depth}")
            return x

        traverse(0, 0)

        trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', hovertext=hover_text,
                           marker=dict(size=12, color=node_y, colorscale='Viridis'), showlegend=False)

        title = f"Decision Tree Visualization (Max Depth={max_depth})" if max_depth else "Decision Tree Visualization (No Max Depth)"
        layout = go.Layout(title=title, xaxis=dict(title='Node Position', showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(title='Depth', showgrid=False, zeroline=False, showticklabels=False),
                           hovermode='closest', plot_bgcolor='white')

        fig = go.Figure(data=[trace], layout=layout)
        model_name = f"decision_tree_max_depth_{max_depth if max_depth is not None else 'None'}.html"
        pio.write_html(fig, file=model_name)
        print(f"Saved visualization to {model_name}")

        dashboard_links.append((max_depth, model_name))
        tree_metrics.append({
            'max_depth': 'None' if max_depth is None else max_depth,
            'n_nodes': n_nodes,
        })

    visualization_links_html = "<ul>"
    for depth, link in dashboard_links:
        depth_str = "No Max Depth" if depth is None else f"Max Depth {depth}"
        visualization_links_html += f'<li><a href="{link}" target="_blank">Decision Tree Visualization ({depth_str})</a></li>\n'
    visualization_links_html += "</ul>"

  
    table_html = """
<table id="treeMetricsTable" class="display" style="width:100%">
    <thead>
        <tr>
            <th>Max Depth</th>
            <th>Number of Nodes</th>
        </tr>
    </thead>
    <tbody>
"""
    for metric in tree_metrics:
        table_html += f"<tr><td>{metric['max_depth']}</td><td>{metric['n_nodes']}</td></tr>\n"
    table_html += """
    </tbody>
</table>
"""

    additional_styles = (
        "/* Center-align table headers */\n"
        "table.dataTable thead th {\n"
        "    text-align: center;\n"
        "}\n"
        "\n"
        "/* Zebra striping for table rows */\n"
        "table.dataTable tr:nth-child(even) {\n"
        "    background-color: #f2f2f2;\n"
        "}\n"
        "\n"
        "/* Hover effect for table rows */\n"
        "table.dataTable tr:hover {\n"
        "    background-color: #ddd;\n"
        "}\n"
        "\n"
        "/* Padding for table headers and cells */\n"
        "table.dataTable th, table.dataTable td {\n"
        "    padding: 12px 15px;\n"
        "}\n"
        "\n"
        "/* Adding some color to the headers */\n"
        "table.dataTable thead th {\n"
        "    background-color: #218d8c;\n"
        "    color: white;\n"
        "}\n"
    )
    
    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Decision Tree Dashboard</title>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.22/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.22/js/jquery.dataTables.js"></script>
        <style>
        {additional_styles}
        </style>
    </head>
    <body>
        <h1>Decision Tree Dashboard</h1>
        {visualization_links_html}
        {table_html}
        <script>
        $(document).ready(function () {{
            $('#treeMetricsTable').DataTable();
        }});
        </script>
    </body>
    </html>
    """

    dashboard_filename = "decision_tree_dashboard.html"
    with open(dashboard_filename, 'w') as f:
        f.write(dashboard_html)

    print(f"Dashboard generated: {dashboard_filename}")
    webbrowser.open('file://' + os.path.realpath(dashboard_filename), new=2)

def create_dfs_by_column(df, col_name):
 
    dfs = {}
    for value in df[col_name].unique():
        dfs[value] = df[df[col_name] == value]
    return dfs




def save_dfs_to_xlsx(dfs_dict, folder_path):
   
    os.makedirs(folder_path, exist_ok=True)
    
    for key, value in dfs_dict.items():

        file_path = os.path.join(folder_path, f"{key}.xlsx")
        
        value.to_excel(file_path, index=False)
        
    print(f"All files have been saved to {folder_path}")




file_path = os.path.expanduser('~/Desktop/News_Articles')

file_name = 'combined_vader_bert_df.xlsx'

combined_vader_bert_df = load_excel_file(file_path, file_name)

X = combined_vader_bert_df['content']

y = combined_vader_bert_df['overall_emotion_vader']

multinomial_nb_model, performance_report = multinomial_nb_trainer(X, y)

combined_vader_bert_multinomial_nb_df = analyze_sentiment_with_multinomial_nb_trainer(combined_vader_bert_df, 'content', multinomial_nb_model)

tfidf_vectorizer = TfidfVectorizer(max_features=1000) 

label_encoder = LabelEncoder()

X = tfidf_vectorizer.fit_transform(combined_vader_bert_multinomial_nb_df['content']).toarray()

y = label_encoder.fit_transform(combined_vader_bert_multinomial_nb_df['overall_emotion_vader'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_and_visualize_decision_trees(X_train, y_train, criterion='gini', max_depths=[None, 10, 5])

lda_model, dt_matrix, topics_df = perform_lda_topic_modeling(combined_vader_bert_df, 'content')

#combined_vader_bert_multinomial_nb_lda_label_df = assign_lda_labels(combined_vader_bert_multinomial_nb_df, topics_df, dt_matrix, label_prefix="Topic")

column_order = ['LDA_label', 'editorial_stance', 'theme', 'emotion_intensity_vader', 'sentiment_score_bert', 'predicted_sentiment_multinomialnb', 'overall_emotion_vader', 'sentiment_category_bert', 'publisher', 'title', 'author', 'publish_date', 'content', 'url']

combined_vader_bert_multinomial_nb_lda_label_df_final =combined_vader_bert_multinomial_nb_df.reindex(columns=column_order)

combined_path = os.path.join(file_path, 'combined_vader_bert_multinomial_nb_lda_label_df.xlsx')

combined_vader_bert_multinomial_nb_lda_label_df_final.to_excel(combined_path, index=False)

dfs_dict = create_dfs_by_column(combined_vader_bert_multinomial_nb_lda_label_df_final, 'editorial_stance')

for key, value in dfs_dict.items():
    globals()[f"df_{key}"] = value
    
    
folder_path = os.path.expanduser('~/Desktop/News_Articles/editorial_stance_dfs')

save_dfs_to_xlsx(dfs_dict, folder_path)

 