# This script is a modification of Kuparinen and Scherrer's (2024) original script.
# Modified by: Ho Wang Matthew Sung
# Last update: June 2025

import matplotlib.pyplot as plt
import json
import re
import numpy as np
import pandas as pd
import os
import glob
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.cluster import homogeneity_completeness_v_measure

from sklearn.metrics.pairwise import cosine_similarity

def topic_model(corpus, inputtype, label, modeltype, no_topics, max_df=1.0, use_idf=False, norm=None, sublinear=False, relevance=False, lambda_=False):                                 
    tfidf_vectorizer = TfidfVectorizer(encoding='utf-8', analyzer='word', max_df=max_df, min_df=2, 
                                    token_pattern=r"(?u)[#_]*\w[\w#_]*[#_]*", lowercase=True, use_idf=use_idf, # the tokenisation pattern is defined specifically for GCND (deals with the morpheme boundary '#')
                                    norm=norm, sublinear_tf=sublinear)
    
    with open("./GCND_full/{}_{}".format(inputtype, corpus), "r", encoding="utf-8") as fp:
        matrix = json.load(fp)
    
    tfidf = tfidf_vectorizer.fit_transform(matrix)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    if modeltype == 'nmf':
        model = NMF(n_components=no_topics, random_state=1).fit(tfidf)
    
    else:
        print("This script only runs NMF. Should you want to apply LDA, please refer to the original code by Kuparinen and Scherrer (2024).")
    
    if relevance == True:
        doc_lengths = tfidf.sum(axis=1).getA1()
        term_freqs = tfidf.sum(axis=0).getA1()
        vocab = tfidf_vectorizer.get_feature_names_out()
        
        def _row_norm(dists):
        # row normalization function required
        # for doc_topic_dists and topic_term_dists
            return dists / dists.sum(axis=1)[:, None]
        
        if norm == None:
            doc_topic_dists = _row_norm(model.transform(tfidf))
            topic_term_dists = _row_norm(model.components_)
            
        else:
            doc_topic_dists = model.transform(tfidf)
            topic_term_dists = model.components_
        
        # compute relevance and top terms for each topic
        term_proportion = term_freqs / term_freqs.sum()
        
        log_lift = np.log(pd.eval("topic_term_dists / term_proportion")).astype("float64")
        log_ttd = np.log(pd.eval("topic_term_dists")).astype("float64")
        
        values_ = lambda_ * log_ttd + (1 - lambda_) * log_lift
        
    else:
        values_ = model.components_
    
    # show top 10 items per topic
    n_words = 10
    topic_model_list = []
    for topic_idx, topic in enumerate(values_):
        top_n = [tfidf_feature_names[i]
                for i in topic.argsort()
                [-n_words:]][::-1]
        top_features = ' '.join(top_n)
        topic_model_list.append(f"topic_{'_'.join(top_n[:3])}") 
        print(f"Topic {topic_idx}: {top_features}")
    
    amounts = model.transform(tfidf)
    
    ### Set it up as a dataframe
    # Find files
    # Set the directory you want to search
    directory = "./GCND_full/corpus"
    # Change to the specified directory
    os.chdir(directory)
    joined_files = os.path.join("*txt")
    joined_list = glob.glob(joined_files)
    metadata = pd.DataFrame(joined_list)
    metadata.rename(columns={metadata.columns[0]: 'DocID'}, inplace=True)
    metadata['DocID'] = [re.sub('.txt', '', sent) for sent in metadata['DocID']]
    
    ### Filenames and dominant topic
    topics = pd.DataFrame(amounts, columns=topic_model_list)
    dominant_topic = np.argmax(topics.values, axis=1)
    topics['dominant_topic'] = dominant_topic
    topics['DocID'] = metadata['DocID'].astype(str)
    
    ### Combine data frames
    model_metadata = pd.merge(metadata, topics, on = "DocID", how = "inner")
    
    ### Save results
    model_metadata.to_csv('./GCND_full/topic_model_data/{}_{}_{}_{}_relevance{}_idf{}_norm{}_sub{}.csv'.format(modeltype, label, no_topics, max_df, lambda_, use_idf, norm, sublinear))

    if relevance == True:
        values_ = np.exp(values_)
        
    ### The plotting function, but save
    def plot_top_words(model, feature_names, n_top_words):
        fig, axes = plt.subplots(1, no_topics, figsize=(25, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(values_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            
            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Component {topic_idx +1}',
                         fontdict={'fontsize': 30})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=25)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
        
        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.savefig('./GCND_full/figures/{}_{}_{}_{}_relevance{}_idf{}_norm{}_sub{}.tiff'.format(modeltype, label, no_topics, max_df, lambda_, use_idf, norm, sublinear), dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig='all')
    
    n_words = 10
    plot_top_words(model, tfidf_feature_names, n_words)
    

    cosine = cosine_similarity(values_)
    np.fill_diagonal(cosine, np.nan)
    cosine = cosine[~np.isnan(cosine)].reshape(cosine.shape[0], cosine.shape[1] - 1)
    max_cosine = cosine.max()
    min_cosine = cosine.min()
    
    print(label, modeltype, no_topics, 'cosine difference', round(max_cosine-min_cosine,2)) # the cosine difference for each topic model


### non-standard, but easier way to run all topic models at once ###
# corpus, format, naming pattern, model, number of components, whether to use inverse document frequency, normalization method, whether to use sublinear term frequency

topic_model('gcnd_full_modified_cleaned', 'trigram', 'trigram_gcnd_full_modified_cleaned', 'nmf', 2, use_idf=True, norm='l2', sublinear=True)
topic_model('gcnd_full_modified_cleaned', 'trigram', 'trigram_gcnd_full_modified_cleaned', 'nmf', 3, use_idf=True, norm='l2', sublinear=True)
topic_model('gcnd_full_modified_cleaned', 'trigram', 'trigram_gcnd_full_modified_cleaned', 'nmf', 4, use_idf=True, norm='l2', sublinear=True)
topic_model('gcnd_full_modified_cleaned', 'trigram', 'trigram_gcnd_full_modified_cleaned', 'nmf', 5, use_idf=True, norm='l2', sublinear=True)

topic_model('gcnd_full_modified_cleaned', 'fourgram', 'fourgram_gcnd_full_modified_cleaned', 'nmf', 2, use_idf=True, norm='l2', sublinear=True)
topic_model('gcnd_full_modified_cleaned', 'fourgram', 'fourgram_gcnd_full_modified_cleaned', 'nmf', 3, use_idf=True, norm='l2', sublinear=True)
topic_model('gcnd_full_modified_cleaned', 'fourgram', 'fourgram_gcnd_full_modified_cleaned', 'nmf', 4, use_idf=True, norm='l2', sublinear=True)
topic_model('gcnd_full_modified_cleaned', 'fourgram', 'fourgram_gcnd_full_modified_cleaned', 'nmf', 5, use_idf=True, norm='l2', sublinear=True)

topic_model('gcnd_full_modified_cleaned', 'words', 'words_gcnd_full_modified_cleaned', 'nmf', 2, use_idf=True, norm='l2', sublinear=True)
topic_model('gcnd_full_modified_cleaned', 'words', 'words_gcnd_full_modified_cleaned', 'nmf', 3, use_idf=True, norm='l2', sublinear=True)
topic_model('gcnd_full_modified_cleaned', 'words', 'words_gcnd_full_modified_cleaned', 'nmf', 4, use_idf=True, norm='l2', sublinear=True)
topic_model('gcnd_full_modified_cleaned', 'words', 'words_gcnd_full_modified_cleaned', 'nmf', 5, use_idf=True, norm='l2', sublinear=True)