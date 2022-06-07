from typing import List, Dict
import datetime
from urllib.parse import urlparse
from torch import nn
from typing import Tuple
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
import re
import scipy

import os
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/shared/siyiliu/covid19-app-old/')



from model.perspectives.perspectrum_model import PerspectrumTransformerModel, MultiTaskBart, ArticleBart, ArgumentMining, QueryBart, CompressorBart

TAG_SET = {
    "source_type": ["Global Agencies", "Government Agencies", "News", "Science", "Academia",0,1,2],

    "doc_type": ["Factual", "Interview", "Medical Publication", "Opinion"],

    "topic": ["News Article","Medical Information","Government Agency", "All Sources", "CNN", "BBC", "Fox", "NYTimes", "Washington Post", "Reuters",
              "ABC", "CBS", "The Guardian", "NPR",  "Nature", "Science News", "CDC", "WHO"]
}
SOURCE_NAME = {
    "www.nature.com": "Nature",
    "www.sciencemag.org": "The Science Magazine",
    "www.nejm.org": "The New England Journal of Medicine",
    "www.yalemedicine.org": "Yale Medicine",
    "www.health.harvard.edu": "Harvard Health Publishing",
    "www.cebm.net": "Centre for Evidence-Based Medicine",
    "www.who.int": "World Health Organization",
    "www.cdc.gov": "Center for Disease Control",
    "health.gov": "U.S. Department of Health and Human Services",
    "www.nytimes.com": "The New York Times",
    "www.washingtonpost.com": "The Washington Post",
    "www.cnn.com": "CNN",
    "www.bbc.com": "BBC",
    "www.reuters.com": "Reuters",
    "www.npr.org": "NPR",
    "www.theguardian.com": "The Guardian",
    "www.sciencenews.org": "Science News",
    "www.foxnews.com" : "Fox News",
    "www.abc.com" : "ABC",
    "www.cbs.com" : "CBS",
    "www.nbcnews.com" : "NBC",
    "www.medicalnewstoday.com": "Medical News Today",
    "www.webmd.com" : "WebMD",
    "www.mayoclinic.org" :"Mayo Clinic",
    "www.medlineplus.gov" : "MedlinePlus",
    "www.familydoctor.org" : "Family Doctor",
    "www.noah-health.org" : "Noah Health",
    "www.nih.gov" : "NIH",
    "www.ninds.nih.gov" : "NIDS NIH",
    "www.clevelandclinic.com": "Cleveland Clinic",
    "www.who.int" : "WHO",
    "www.nejm.org" :"NEJM",
    "www.yalemedicine.org" : "Yale Medicine"
}

SOURCE_TYPE = {
    "www.nature.com": "Science",
    "www.sciencemag.org": "Science",
    "www.nejm.org": "Medical Information",
    "www.yalemedicine.org": "Medical Information",
    "www.health.harvard.edu": "Medical Information",
    "www.cebm.net": "Medical Information",
    "www.who.int": "Medical Information",
    "www.cdc.gov": "Medical Information",
    "health.gov": "Medical Information",
    "www.nytimes.com": "News Article",
    "www.washingtonpost.com": "News Article",
    "www.cnn.com": "News Article",
    "www.bbc.com": "News Article",
    "www.reuters.com": "News Article",
    "www.npr.org": "News Article",
    "www.theguardian.com": "News Article",
    "www.sciencenews.org": "News Article",
    "www.foxnews.com" : "News Article",
    "www.abc.com" : "News Article",
    "www.cbs.com" : "News Article",
    "www.nbcnews.com" : "News Article",
    "www.medicalnewstoday.com": "Medical Information",
    "www.webmd.com" : "Medical Information",
    "www.mayoclinic.org" :"Medical Information",
    "www.medlineplus.gov" : "Medical Information",
    "www.familydoctor.org" : "Medical Information",
    "www.noah-health.org" : "Medical Information",
    "www.nih.gov" : "Medical Information",
    "www.ninds.nih.gov" : "Medical Information",
    "www.clevelandclinic.com": "Medical Information",
    "www.who.int" : "Medical Information",
    "www.nejm.org" :"Medical Information",
    "www.yalemedicine.org" : "Medical Information"
}

# Load topic classification model
TOPIC_MODEL_DIR = "model/topic_classifier/"
with open(os.path.join(TOPIC_MODEL_DIR, "vectorizer.p"), "rb") as fin:
    vectorizer = pickle.load(fin)

with open(os.path.join(TOPIC_MODEL_DIR, "xtrain.p"), "rb") as fin:
    xtrain = pickle.load(fin)

with open(os.path.join(TOPIC_MODEL_DIR, "label.p"), "rb") as fin:
    id_label = pickle.load(fin)
    
print(id_label)


# Load relevance and stance models
PERSPECTIVES_MODEL_DIR = "model/perspectives/"
stance_model = PerspectrumTransformerModel("roberta", PERSPECTIVES_MODEL_DIR + "stance")
relevance_model = PerspectrumTransformerModel("roberta", PERSPECTIVES_MODEL_DIR + "relevance")

BART_DIR = "model/perspectives/MultiTaskBart"
summarize_model = MultiTaskBart(BART_DIR)


Q_BART_DIR = "model/perspectives/Bart_with_query"
summarize_model_with_query = QueryBart(Q_BART_DIR)

#Article_BART_DIR = "model/perspectives/article_summarization_model"
#article_summarize_model = ArticleBart(Article_BART_DIR)

AM_DIR = "model/perspectives/AM_IBM_wiki"
argument_mining_model = ArgumentMining(AM_DIR)

Compressor_DIR = "model/perspectives/CompressorBart"
Compressor_model = CompressorBart(Compressor_DIR)



class Article:
    def __init__(self,
                 url: str,
                 text: str,
                 title: str = None,
                 top_img_url: str = None,
                 publish_date: datetime.datetime = None,
                 dimensions: Dict[str, List[str]] = None,
                 authors: List[str] = None,
                 links: List[str] = None,
                 people: List[str] = None,
                 orgs: List[str] = None,
                 quotes: List[tuple] = None,
                 **kwargs) -> None:
        """

        :param url:
        :param text:
        :param dimensions: A dictionary storing [dimension name -> List of tags on this dimension]
        :param links: A list of all links in article
        :param people: A list of unique named entities of type PERSON in the article
        :param orgs: A list of unique named entities of type ORG in the article
        :param quotes: A list of all quotes from stanford quote annotator model
        :param kwargs:
        """
        self.title = title
        self.url = url
        self.text = text
        # Manual override of WHO picture, since default is pixelated.
        if urlparse(url).netloc == "www.who.int":
            self.top_img_url = "https://upload.wikimedia.org/wikipedia/commons/5/51/Who-logo.jpg"
        else:
            self.top_img_url = top_img_url
        self.quotes = quotes
        if publish_date is not None:
            self.publish_date = publish_date.date()
        else:
            self.publish_date = None

        if authors is not None:
            self.authors = authors
        else:
            self.authors = []

        self.links = links
        self.people = people
        self.orgs = orgs
        self.quotes = quotes

        if dimensions is not None:
            self.dimensions = dimensions
        else:
            self.dimensions = {}

        _url_host = urlparse(url).netloc
        if _url_host in SOURCE_NAME:
            self.source_name = SOURCE_NAME[_url_host]
        else:
            self.source_name = _url_host


def classify_document_type(article: Article) -> List[str]:
    """
    (Multiclass) classifier that assigns document type labels

    TODO: Make a batched version of this function?
    :param article: an input article
    :return: A list of predicted article type tags for the input article
    """
    _src_type = classify_document_source_type(article)[0]
    return [_src_type]
#    _doc_type = "Other"
#    if _src_type == "News":
#        _doc_type = "News Article"
#    elif _src_type == "Global Agencies" or _src_type == "Government Agencies":
#        _doc_type = "Factual"
#    elif _src_type == "Academia":
#        _doc_type = "Medical Information"
#    return [_doc_type]


def classify_document_source_type(article: Article) -> List[str]:
    """
    (Multiclass) classifier that predicts the type of source of a document
    :param article: an input article
    :return: A list of predicted article source type tags for the input article
    """
    _url = article.url
    _host = urlparse(_url).netloc

    if _host in SOURCE_TYPE:
        return [SOURCE_TYPE[_host]]
    else:
        print(_host)
        return ["Other"] # TODO: should never happen...


def classify_document_topic(article: Article,
                            k:int = 2) -> List[str]:
    """
    (Multiclass) classifier that predicts the type of source of a document
    :param article: an input article
    :param k: number of labels to return
    :return: A list of topic tags
    """

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=1)

    X_test = vectorizer.transform([article.text])
    cos_sim = cosine_similarity(X_test, xtrain)
    predicted_score = softmax(cos_sim)
    predicted_labels = []
    for sid, score in enumerate(predicted_score):
        predicted_labels = score.argsort()[-k:][::-1]
    return_labels = []
    for l in predicted_labels:
        return_labels.append(id_label[l])
    return return_labels


def classify_relevant_documents(article_candidates: List[Article]) -> List[Article]:
    """
    Given a list of article candidates (retrieved by search engine), identify the relevant subset that we want to show
    :param article_candidates:
    :return:
    """
    return article_candidates


def get_relevant_dimensions_for_query(query: str) -> List[str]:
    """
    Given a user query, return a list of “dimensions” we would like to show with this query

    TODO: I'm actually not sure what exactly this one should do... Maybe think about this later?
    :param query:
    :return:
    """
    pass


def get_relevant_sentences(query: str, urls: List[str], articles: Dict, max_sentence_length=128):
    """
        Get relevant sentences with respect to a claim for each article returned
        :param query:
        :param urls:
        :param articles
        :param max_sentence_length
        :return: A dictionary of relevant sentences for each article
    """

    threshold = -3
    relevant_sentences = {url: [] for url in urls}
    for url in urls:
        article = articles[url]
        paragraphs = [p for p in article.text.splitlines() if p]
        if paragraphs:
            sents = [sent_tokenize(paragraph) for paragraph in paragraphs]
            query_trunc = query[:max_sentence_length]
            pairs = [(query_trunc, sent[0][:max_sentence_length]) for sent in sents]
            #print("relevant", pairs)
            predictions = relevance_model.predict_batch(pairs)
            for i in range(len(pairs)):
                if predictions[i][1] >= threshold:
                    if len(sents[i]) > 1:
                        description = " ".join(sents[i][1:])
                    else:
                        description = ""
                    relevant_sentences[url] += [(sents[i][0], predictions[i][1], description)]
    return relevant_sentences


def get_main_perspective_and_description(relevant_sentences: List[str], article: Article):
    """
    Get the most relevant perspective in an article w.r.t the user query
    :param relevant_sentences
    :param article:
    :return: The most relevant sentence/perspective in article, its score, and a description
    """

    # Currently takes sentence with largest relevance score
    max = -3
    max_sent = ""
    desc = ""
    for sent in relevant_sentences:
        if sent[1] >= max:
            max_sent = sent[0]
            max = sent[1]
            desc = sent[2]

    if desc == "":
        paragraphs = [p for p in article.text.splitlines() if p]
        if paragraphs:
            desc = paragraphs[0]

    return max_sent, max, desc


def get_equivalent_clusters(query: str, relevant_quotes: Dict, urls: List[str], K=3, n_clusters=3):
    """
        Get set of equivalent articles for each article
        :param query:
        :param relevant_quotes:
        :param urls
        :param K:
        :param n_clusters
        :return: The articles equivalent to each other article
    """
    equivalent_articles = {url: [] for url in urls}
    quotes = top_k_quotes(relevant_quotes, K)
    embeddings = []
    relevant_articles = []
    for url in urls:
        sents = [(query, quote[0]) for quote in quotes[url]]
        print(sents[0])
        weights = np.array([quote[1] for quote in quotes[url]])
        num = len(sents)
        CLS_embeddings = stance_model.batch_CLS_embeddings(sents)
        #print(np.array(CLS_embeddings).shape)
        if len(sents) != 0:
            relevant_articles += [url]
            embedding = np.mean(np.array(CLS_embeddings) * weights.reshape(num, -1), axis=0)
            embedding /= np.linalg.norm(embedding)
            embeddings += [embedding]

    #print(len(embeddings))
    #print(embeddings[0].shape)
    #print(relevant_articles)
    
    final_number_cluster = n_clusters if n_clusters <= len(embeddings) else len(embeddings)
    clusters = KMeans(final_number_cluster).fit(embeddings).labels_

    # Get all articles equivalent to every other article
    for i in range(len(relevant_articles)):
        num = clusters[i]
        equivalent_articles[relevant_articles[i]] = \
            [relevant_articles[j] for j, x in enumerate(clusters) if x == num and j != i]

    url_clusters = []
    for i in range(n_clusters):
        url_clusters += [[relevant_articles[j] for j, x in enumerate(clusters) if x == i]]

    return equivalent_articles, url_clusters


# Helper method for get_equivalent_clusters
def top_k_quotes(relevant_quotes, K):
    max_quotes = {}
    urls = list(relevant_quotes.keys())
    for url in urls:
        quotes = relevant_quotes[url]
        if len(quotes) == 0:
            max_quotes[url] = []
        else:
            scores = np.array([tup[1] for tup in quotes])
            max_args = scores.argsort()[-K:][::-1]
            max_quotes[url] = [quotes[arg] for arg in max_args]
    return max_quotes


def get_related_articles(equivalent_articles: List[str], articles: Dict) -> List:
    related_articles = []
    for url in equivalent_articles:
        article = articles[url]
        related_articles += [{
            "url": article.url,
            "img_url": article.top_img_url,
            "title": article.title,
            "category": "Shares perspective"
        }]
    return related_articles


def format_query(query_text):
    """
        Formats a user's query as a title.

        TODO: If query is in the form of a question, convert it to a claim
        :param query_text Raw user input to search bar
        :return: Formatted input capitalized as a title
    """
    words = word_tokenize(query_text)
    tagged_words = pos_tag([word.lower() for word in words])
    capitalized_words = [w.capitalize() if t not in ["IN"] else w for (w, t) in tagged_words]
    capitalized_words[0] = capitalized_words[0].capitalize()
    capitalized_acronyms = []
    for i in range(len(capitalized_words)):
        w = capitalized_words[i]
        if w.lower() in ["u.s.", "usa", "covid-19", "covid", "u.s.a"]:
            capitalized_acronyms += [w.upper()]
        elif w.lower() in ["us"] and tagged_words[i][1] in ["NNP"]:  # This still doesn't work as well as I'd like it to
            capitalized_acronyms += [w.upper()]
        else:
            capitalized_acronyms += [w]
    text_truecase = re.sub(" (?=[.,'!?:;])", "", ' '.join(capitalized_acronyms))
    return text_truecase

def get_summarization(model, paragraph, query=None):
    if query==None:
        return model.summarize_one(paragraph)
    else:
        return model.summarize_one(paragraph, query)

def get_article_summarization(paragraph):
    return article_summarize_model.summarize_one(paragraph)
    
def get_arguments_from_article(topic, article):
    return argument_mining_model.get_arguments(topic, article)

def get_compression(sentence: str):
    return Compressor_model.compress_one(sentence)

def get_relevance_score(sent1,sent2):
    return relevance_model.predict_batch[sent1,sent2]

def shorten_paragraph(paragraphs, length):
    """
    :param pragraphs A list of paragraphs to be shorten
    :param length The number of tokens we want it to be shortened to
    """
    res = ""
    for para in paragraphs:
        sents = sent_tokenize(para)
        for sent in sents:
            if len(word_tokenize(res))<=length:
                res = res + ' ' + sent
            else:
                return res
    return res
    
    
#    str_paragraphs = ' '.join(paragraphs)
#    sent_lst = sent_tokenize(str_paragraphs)
#    res = ''
#    for sent in sent_lst:
#        if len(sent)>2:
#            temp = res + sent + '. '
#            if len(temp.split(' '))<length:
#                res = temp
#            else:
#                break
#    return res
    
def shorten_relevant_sentences(relevant_sentences, length):
    """
    Shorten the relevant sentences and rank by their score
    :param relevant_sentences
    :param length The number of tokens we want it to be shortened to
    """
    # not using this any more since we are summarizing using the first several paragraphs
    sorted_lst = sorted(relevant_sentences, key = lambda x: x[1])
    res = ''
    for sent in sorted_lst:
        if len(sent[0])>3:
            temp = res + sent[0] + ' '
            if len(temp.split(' '))<length:
                res = temp
            else:
                break
    return res
        
        
def get_equivalent_clusters_from_titles(query: str, titles: List[str], urls: List[str], K=3, n_clusters=3):
    """
        Get set of equivalent articles for each article
        :param query:
        :param relevant_quotes:
        :param urls
        :param K:
        :param n_clusters
        :return: The articles equivalent to each other article
    """
    equivalent_articles = {url: [] for url in urls}
    
    embeddings = []
    relevant_articles = []
    for i,url in enumerate(urls):
        sents = [(query, titles[i][0])]
        num = len(sents)
        CLS_embeddings = stance_model.batch_CLS_embeddings(sents)
        if len(sents) != 0:
            relevant_articles += [url]
            embeddings += [np.array(CLS_embeddings[0])]
            
    #print(len(embeddings))
    #print(embeddings[0].shape)
    #print(relevant_articles)

    final_number_cluster = n_clusters if n_clusters <= len(embeddings) else len(embeddings)
    print("embeddings here", embeddings)
    kmeans = KMeans(final_number_cluster).fit(embeddings)
    #kmeans = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=20).fit(embeddings)
    
#    for eps in range(30,10,-1):
#        kmeans = DBSCAN(eps=eps,min_samples=2).fit(embeddings)
#        clusters= kmeans.labels_
#        print(clusters)
#        if max(clusters)>=2:
#            break
    clusters= kmeans.labels_
    #print('eps=', eps)
    
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
    #print('closest', closest)
    closest_to_centroids= closest #kmeans.core_sample_indices_

    # Get all articles equivalent to every other article
    for i in range(len(relevant_articles)):
        num = clusters[i]
        equivalent_articles[relevant_articles[i]] = \
            [relevant_articles[j] for j, x in enumerate(clusters) if x == num and j != i]

    # this part is different than the original get_clusters function
    
    url_clusters = []
    for i in range(n_clusters):
        url_clusters += [[relevant_articles[j] for j, x in enumerate(clusters) if x == i]]
        
    #url_clusters = {}
    #for i, url in enumerate(relevant_articles):
    #    url_clusters[url] = clusters[i]

    return equivalent_articles, url_clusters, closest_to_centroids
    
def get_equivalent_clusters_in_stance(query: str, perspectives: List[str], urls: List[str]):
    threshold_pos = 2
    threshold_neg = -2
    cluster_lst = []
    pairs = [(query, persp[0]) for persp in perspectives]
    #print(pairs)
    center_perspectives=[]
    predictions = stance_model.predict_batch(pairs)
    predictions = [list(a)for a in predictions]
    for i in range(len(pairs)):
    
        if predictions[i][1] >threshold_pos and predictions[i][0] <threshold_neg:
            cluster_lst.append(0)
        elif predictions[i][0] >threshold_pos and predictions[i][1] <threshold_neg:
            cluster_lst.append(1)
        else:
            cluster_lst.append(2)
    print("clusters", cluster_lst)
    print(predictions)
    max_ = predictions.index(max(predictions, key=lambda x:x[1]))
    min_ = predictions.index(max(predictions, key=lambda x:x[0]))
    mid_ = predictions.index(min(predictions, key=lambda x:abs(x[1]-0)))
    
#    if cluster_lst.count(0)==0:
#        if (cluster_lst[max_] ==2 and cluster_lst.count(2) ==1) or cluster_lst.count(2)==0:
#            temp = predictions
#            temp[max_] = (0,-10)
#            cluster_lst[temp.index(max(temp, key=lambda x:x[1]))]=2
#            cluster_lst[max_] =0
#        else:
#            cluster_lst[max_] =0
#        #cluster_lst[predictions.index(max(predictions, key=lambda x:x[1]))]=0
#    elif cluster_lst.count(1)==0:
#        if (cluster_lst[min_] ==2 and cluster_lst.count(2) ==1) or cluster_lst.count(2)==0:
#            temp = predictions
#            temp[min_] = (-10,0)
#            cluster_lst[temp.index(max(temp, key=lambda x:x[0]))]=2
#            cluster_lst[min_] =1
#        else:
#            cluster_lst[min_] =1
    
        #cluster_lst[predictions.index(max(predictions, key=lambda x:x[0]))]=1
#    if cluster_lst.count(2)==0:
#        print(max_)
#        print(min_)
#        print(mid_)
#        if mid_ == max_ and cluster_lst.count(0)<=1:
#            temp = predictions
#            temp[max_] = (0,-10)
#            cluster_lst[temp.index(max(temp, key=lambda x:x[1]))]=2
#            #cluster_lst[max_] =0
#        elif mid_ == min_ and cluster_lst.count(1)<=1:
#            temp = predictions
#            temp[min_] = (-10,0)
#            cluster_lst[temp.index(max(temp, key=lambda x:x[0]))]=2
#        else:
#            cluster_lst[mid_] = 2
        #cluster_lst[predictions.index(min(predictions, key=lambda x:abs(x[1]-0)))]=2
#    print(cluster_lst)
    
    #relevance
    pairs_rel = [(query, persp[0]) for persp in perspectives]
    pred_rel = relevance_model.predict_batch(pairs_rel)
    pred_rel = [list(a)for a in pred_rel]
    
    max_pos = -10
    max_pos_ind = -1
    max_neg = -10
    max_neg_ind = -1
    max_mid = -10
    max_mid_ind = -1
    
    print("rel preds", pred_rel)
    print("perspectives", perspectives)
    for ind, each in enumerate(cluster_lst):
        
        if each == 0:
            if pred_rel[ind][1] >max_pos:
                max_pos = pred_rel[ind][1]
                max_pos_ind = ind
                
        if each == 1:
            if pred_rel[ind][1] >max_neg:
                max_neg = pred_rel[ind][1]
                max_neg_ind = ind
            
        if each == 2:
            if pred_rel[ind][1] >max_mid:
                max_mid = pred_rel[ind][1]
                max_mid_ind = ind
                
    
    center_perspectives.append(max_pos_ind)
    center_perspectives.append(max_neg_ind)
    center_perspectives.append(max_mid_ind)

    #print(predictions)
    #print(scipy.special.softmax(predictions[0]))
    #center_perspectives.append(predictions.index(max(predictions, key=lambda x:x[1])))
    #center_perspectives.append(predictions.index(max(predictions, key=lambda x:x[0])))
    #center_perspectives.append(predictions.index(min(predictions, key=lambda x:abs(x[1]-0))))
    
    
    url_clusters = []
    for i in range(3):
        url_clusters += [[urls[j] for j, x in enumerate(cluster_lst) if x == i]]
    
    #print('url_clusters!!!', url_clusters)
    return url_clusters, center_perspectives
    

def get_perspectives(articles, query=None):
    #for now the model does not incorporatate the query. Need to update our query dependent summarization model.
    perspective_lst=[]
    perspective_dic = {}
    
    if query ==None:
        for url in articles:
            # Here we parse each HTML page into Article class, with cleaner text + metadata we want
            article = articles[url]
            
            # if we wanna summarize using the first paragraph
            paragraphs = [p for p in article.text.splitlines() if p]

           

            perspective =get_summarization(summarize_model, [shorten_paragraph(paragraphs, 200)])
            #print("Perspective2: ", summary2[0])
            perspective_lst.append(perspective)
            perspective_dic[url] = perspective
    else:
        for url in articles:
            article = articles[url]
            paragraphs = [p for p in article.text.splitlines() if p]
            
            perspective =get_summarization(summarize_model_with_query, [shorten_paragraph(paragraphs, 200)], query=[query])
            #print("Perspective2: ", summary2[0])
            perspective_lst.append(perspective)
            perspective_dic[url] = perspective
        
        
        
    return perspective_lst, perspective_dic
    
def get_arguments(article, query, confidence_threshold=0.5):

    #print("!!!!!!!!!!!!!")
    #print("title", article.title)
    #print()
    #print(article.url)
    paragraphs = [p for p in article.text.splitlines() if p]
    #print("text", paragraphs)
    #print("!!!!!!!!!!!!!")
    sentences = []
    for para in paragraphs:
        if len(sent_tokenize(para))<=500:
            sentences += sent_tokenize(para)
        #print(sent)
        
    arguments= []
    #print(query, sentences)
    arguments_prob = get_arguments_from_article(query, sentences)
    arguments_prob = [each[0] for each in arguments_prob]
    for i in range(len(arguments_prob)):
        #print("score=", float(arguments_prob[i][0]))
        if float(arguments_prob[i])>=confidence_threshold:
            arguments.append((sentences[i],arguments_prob[i]))
    #print(arguments)
    if len(arguments)>5:
        arguments.sort(key = lambda x: x[1], reverse=True)
        return arguments[:5]
    else:
        combined = list(zip(sentences,arguments_prob))
        combined.sort(key = lambda x: x[1], reverse=True)
        return combined[:5]
        
def get_key_arguments(perspective: str, arguments: List[Tuple[str,float]]):
    print("Args", arguments)
    assert type(perspective) == str
    assert type(arguments[0][0]) == str
    
    
    compressed_arguments=[get_compression(arg[0]) for arg in arguments]
    
    pairs_stance = [(perspective, arg) for arg in compressed_arguments]
    pred_stance = stance_model.predict_batch(pairs_stance)
    pred_stance = [list(a)for a in pred_stance]
    print("stance score", pred_stance)
    zipped = list(zip(range(0,len(compressed_arguments)), pred_stance))
    print(zipped)
    zipped.sort(key = lambda x: x[1][1], reverse=True)
    print(zipped)
    stance_correct = zipped[:3]
    
    
    
    #print("compressed", compressed_arguments)
    
    
    if len([each for each in stance_correct if each[1][1]>0])<=1:
        key_persp = compressed_arguments[stance_correct[0][0]]
        print("pos number1", len([each for each in stance_correct if each[1][1]>0]))
    else:
        stance_correct_positive = [each for each in stance_correct if each[1][1]>0]
        stance_correct_pos_compressed = [compressed_arguments[each[0]] for each in stance_correct_positive]
        pairs_rel = [(perspective, arg) for arg in stance_correct_pos_compressed]
        pred_rel = relevance_model.predict_batch(pairs_rel)
        pred_rel = [list(a)for a in pred_rel]
        key_persp =stance_correct_pos_compressed[pred_rel.index(max(pred_rel, key=lambda x:x[1]))]
        print("pos number2", len([each for each in stance_correct if each[1][1]>0]))
    
    #print(pred_rel)
    return [arguments[each[0]] for each in stance_correct], key_persp
    
    
    

def filter_articles_by_length(articles, urls):
    filtered_articles = {}
    filtered_urls = []
    for url in articles:
        article=articles[url]
        paragraphs = [p for p in article.text.splitlines() if p]
        if paragraphs!=[]:
            sentences = []
            for para in paragraphs:
                sentences += sent_tokenize(para)
                #print(sent)
                            
            if len(sentences) >=3:
                filtered_articles[url] = article
                filtered_urls.append(url)
    return filtered_articles, filtered_urls
            
def filter_articles_by_perspectives_relevance(articles, urls, perspectives, query):
    filtered_articles = {}
    filtered_urls = []
    filtered_perspectives= []
    print("perspectives:", perspectives)
    
    pairs_rel = [(query, persp[0]) for persp in perspectives]
    pred_rel = relevance_model.predict_batch(pairs_rel)
    pred_rel = [list(a)for a in pred_rel]
    
    for ind, url in enumerate(urls):
        
        if pred_rel[ind][1]>0:
            filtered_articles[url] = articles[url]
            filtered_urls.append(url)
            filtered_perspectives.append(perspectives[ind])
    
    print("relpreds, ", pred_rel)
    
    return filtered_articles, filtered_urls, filtered_perspectives
    
def get_date_range(dates):

    print(dates)

    if len(dates)<=1:
        return dates[0]

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    lst = []
    for date in dates:
        date_lst = date.split(' ')
        print(date_lst)
        cur_month = [months.index(month) for month in months if month in date_lst[0]][0]
        cur_year = [each for each in date_lst if each.isdigit()][0]
        lst.append([cur_year,cur_month])
        
    lst = sorted(lst, key=lambda x: (x[0], x[1]))
    print(lst)
    
    str_ =  months[lst[0][1]] + ' '+ str(lst[0][0]) + ' - ' + months[lst[-1][1]] + ' '+ str(lst[-1][0])
                    
    return str_
if __name__ == '__main__':
    
    #doc = Article(url="dummy", text="mask icon Use of Cloth Face Coverings to Help Slow the Spread of COVID-19\nCDC continues to study the spread and effects of the novel coronavirus across the United States. We now know from recent studies that a significant portion of individuals with coronavirus lack symptoms (“asymptomatic”) and that even those who eventually develop symptoms (“pre-symptomatic”) can transmit the virus to others before showing symptoms. This means that the virus can spread between people interacting in close proximity—for example, speaking, coughing, or sneezing—even if those people are not exhibiting symptoms. In light of this new evidence, CDC recommends wearing cloth face coverings in public settings where other social distancing measures are difficult to maintain (e.g., grocery stores and pharmacies) especially in areas of significant community-based transmission.\nIt is critical to emphasize that maintaining 6-feet social distancing remains important to slowing the spread of the virus. CDC is additionally advising the use of simple cloth face coverings to slow the spread of the virus and help people who may have the virus and do not know it from transmitting it to others. Cloth face coverings fashioned from household items or made at home from common materials at low cost can be used as an additional, voluntary public health measure.\nThe cloth face coverings recommended are not surgical masks or N-95 respirators. Those are critical supplies that must continue to be reserved for healthcare workers and other medical first responders, as recommended by current CDC guidance.\nThis recommendation complements and does not replace the President’s Coronavirus Guidelines for America, 30 Days to Slow the Spreadexternal icon, which remains the cornerstone of our national effort to slow the spread of the coronavirus. CDC will make additional recommendations as the evidence regarding appropriate public health measures continues to develop.")
    #print(classify_document_topic(doc))
    #shorten_paragraph(["The Centers for Disease Control and Prevention said on Monday about 184.4 million people have received at least one dose of a Covid-19 vaccine, including about 159.5 million people who have been fully vaccinated by Johnson & Johnson’s single-dose vaccine or the two-dose series made by Pfizer-BioNTech and Moderna", "About 67.7 percent of adults have received at least one shot. President Biden set a goal on May 4 of reaching 70 percent of adults by July 4, but has since acknowledged the country would need additional time to achieve the national target. Here’s how states are progressing towards the 70 percent benchmark.", "Providers are administering about 0.57 million doses per day on average, about a 83 percent decrease from the peak of 3.38 million reported on April 13.", "Figures show the date shots were reported, rather than the date shots were given and include first and second doses of Pfizer-BioNTech and Moderna, and single doses of Johnson & Johnson."],200)
    get_equivalent_clusters_in_stance("should we still practice social distancing?", [["We should practice social distancing"]], ["http"])
    
