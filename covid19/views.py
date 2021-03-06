import sys
import json
import copy
import pickle
from collections import OrderedDict

from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

from covid19.search.google_custom_search import CustomSearchClient, get_urls_from_search_results, get_dates_from_search_results, get_snippets_from_search_results, get_eval_data
from covid19.search.news_html_parser import parse_article
from covid19.apis import *
from covid19.models import SearchCache
import torch
import csv

# Config file, loaded during server start
# Look at covid19/config/config.json for more details

if not os.path.exists(settings.COVID19_APP_CONFIG_PATH):
    print("Please create the configuration file at {}. Refer to the dev repo readme for more details. "
          .format(settings.COVID19_APP_CONFIG_PATH), file=sys.stderr)
    exit(1)



config = json.load(open(settings.COVID19_APP_CONFIG_PATH))


def render_home_page(request):
    """
    Rendering homepage for the COVID information app
    :param request:
    :return:
    """
    context = {}
    return render(request, 'index.html', context)


def render_about(request):
    """
    Rendering homepage for the COVID information app
    :param request:
    :return:
    """
    context = {}
    return render(request, 'about.html', context)


def render_query_results(request):
    query_text = request.GET.get('q', "")

    # Issue a query to google and get results
    google_search_client = CustomSearchClient(key=config["custom_search_api_key"], cx=config["custom_search_engine_id"])
    google_results = google_search_client.query(query_text)
    _urls = get_urls_from_search_results(google_results)
    
    articles = {url: parse_article(url) for url in _urls}
    relevant_sentences = get_relevant_sentences(query_text, _urls, articles)

    article_list_by_source = {}
    
    titles=[]
    for i in _urls:
        titles.append(articles[i].title)
        
    print(titles)
    #equivalent_clusters, url_batches = get_equivalent_clusters_from_titles(query_text, titles, _urls)
    
    
    perspectives_lst = []
    arguments_lst_by_article ={}
    for url in _urls:
        # Here we parse each HTML page into Article class, with cleaner text + metadata we want
        article = parse_article(url)
        sentences_lst = relevant_sentences[url]
        relevant_sentences_to_sum = [sentences_lst[i][0] for i in range(len(sentences_lst))]
        
        #print('-----Start-----')
        #print()
        #print('Relevant Sentences: ',' '.join(relevant_sentences_to_sum))
        #print()
        #print('Rank Relevant: ', shorten_relevant_sentences(sentences_lst, 200))
        #print()
        #summary =get_summarization([shorten_relevant_sentences(sentences_lst, 200)])
        #print("Perspective1: ", summary[0])
        #print()
        #article.perspective = summary[0]
        
        
        # if we wanna summarize using the first paragraph
        paragraphs = [p for p in article.text.splitlines() if p]
        #print("First Paragraph: ", paragraphs[0])
        #print(paragraphs)
        sentences = []
        for para in paragraphs:
            sent = para.split('. ')
            #print(sent)
            if len(sent)>1:
                for each in sent:
                    if len(each)>5:
                        sentences += [each]
        #print("sent",sentences)
        #print("Shortened Paragraph: ", shorten_paragraph(paragraphs[:3], 200))
        print()
        summary2 =get_summarization([shorten_paragraph(paragraphs[:3], 200)])
        #print("Perspective2: ", summary2[0])
        article.perspective2 = summary2[0]
        perspectives_lst.append(summary2[0])
        #print()
        
        print(article.title)
        print("Arguments:")
        arguments= []
        arguments_prob = get_arguments_from_article(query_text, sentences)
        for i in range(len(arguments_prob)):
            #print("score=", float(arguments_prob[i][0]))
            if float(arguments_prob[i][0])>=0.5:
                arguments.append((sentences[i],arguments_prob[i][0]))
        #print(arguments)
        if len(arguments)>3:
            arguments.sort(key = lambda x: x[1], reverse=True)
            print(arguments[:3])
        else:
            combined = list(zip(sentences,arguments_prob))
            combined.sort(key = lambda x: x[1], reverse=True)
            print(combined[:3])
        print()
        #print("Sentences:")
        #print(sentences)
                
        
        #print("Article: ", shorten_paragraph(paragraphs, 1000))
        #summary3 = get_article_summarization([shorten_paragraph(paragraphs, 600)])
        #print("Perspective3: ", summary3[0])
        #article.perspective3 = summary3[0][:-1]
        #print('-----End------')
        #print()
        
        #print(article.title)
        #print(url_batches[url])
        # We call each classifier to assign categorical metadata to the articles
        _source_type =  classify_document_source_type(article)[0] #url_batches[url]
        #print('source type')
        #print(_source_type)
        
        # article.dimensions["source_type"] = _source_type
        article.dimensions["doc_type"] = classify_document_type(article)
        article.dimensions["topic"] = classify_document_topic(article)

        if _source_type not in article_list_by_source:
            article_list_by_source[_source_type] = [article]
        else:
            article_list_by_source[_source_type].append(article)
            
    #print(OrderedDict(sorted(article_list_by_source.items(), key=lambda x: TAG_SET["source_type"].index(x[0]))))
    #print(article_list_by_source.items())
    #print(perspectives_lst)
    equivalent_clusters, url_batches, closest_to_centroids = get_equivalent_clusters_from_titles(query_text, perspectives_lst, _urls)
    
    for i in url_batches:
        print(articles[i].title)
        print(url_batches[i])
    for i in closest_to_centroids:
        print(perspectives_lst[i])
    
    with open('{}_dbscan.txt'.format(query_text), 'w') as file:
        for i, url in enumerate(url_batches):
            print(perspectives_lst[i], articles[url].title)
            file.write(perspectives_lst[i] + '\t' + str(url_batches[url]) + '\t' + articles[url].title + '\n')
        

    
    article_list_by_source = OrderedDict(sorted(article_list_by_source.items(), key=lambda x: TAG_SET["source_type"].index(x[0])))
    context = {
        "query_text": query_text,
        "article_list_by_source": article_list_by_source,
        "source_types": TAG_SET["source_type"],
        "document_types": TAG_SET["doc_type"],
        "topics": TAG_SET["topic"]
    }

    return render(request, 'search_result.html', context)


def render_quote(request):
    quote = request.GET.get('q', "")

    # For the moment, I'm just searching google for quote. This should be improved upon in the future...
    # Issue a query to google and get results
    google_search_client = CustomSearchClient(key=config["custom_search_api_key"], cx=config["custom_search_engine_id"])
    google_results = google_search_client.query(quote)
    _urls = get_urls_from_search_results(google_results)

    article_list_by_source = {}

    for url in _urls:
        # Here we parse each HTML page into Article class, with cleaner text + metadata we want
        article = parse_article(url)

        # We call each classifier to assign categorical metadata to the articles
        _source_type = classify_document_source_type(article)[0]
        # article.dimensions["source_type"] = _source_type
        article.dimensions["Document Type"] = classify_document_type(article)
        article.dimensions["Topic"] = classify_document_topic(article)

        if _source_type not in article_list_by_source:
            article_list_by_source[_source_type] = [article]
        else:
            article_list_by_source[_source_type].append(article)

    context = {
        "quote": quote,
        "article_list_by_source": article_list_by_source,
        "source_types": TAG_SET["source_type"],
        "document_types": TAG_SET["doc_type"],
        "topics": TAG_SET["topic"]
    }
    return render(request, 'quote.html', context)


def _apply_tags(full_response: Dict,
                target_tags: List):
    """
    filter a full response by a set of target_tags. If an article contains none of the target tags, drop it.
    :param full_response:
    :return: a filtered response with tags applied. full response is returned when no tags is provided
    """
    if not target_tags:
        full_response['tags'] = []
        return [full_response]

    target_tags = set(target_tags)
    filtered_resp = {key: full_response[key] for key in ["title", "tags"]}
    filtered_resp["perspectives"] = []

    for pid, persp in enumerate(full_response["perspectives"]):
        current_perspective = {
            "articles": [],
            "sources": []
        }
        for aid, atc in enumerate(persp["articles"]):

            current_tags_set = set(atc["article_topics"] + atc["article_type"])
            u_tags = current_tags_set.intersection(target_tags)

            if len(u_tags) > 0:
                current_perspective["articles"].append(atc)
                current_perspective["sources"].append(atc["source"])

                if "main_perspective" not in current_perspective:
                    current_perspective["main_perspective"] = persp["main_perspective"]
                    current_perspective["description"] = persp["description"]

        if len(current_perspective["articles"]) > 0:
            current_perspective["sources"] = list(set(current_perspective["sources"]))
            filtered_resp["perspectives"].append(current_perspective)

    if len(filtered_resp["perspectives"]) == 0:
        filtered_resp = []
        print("Here!!!! Tags: ", target_tags,full_response['tags'])
    else:
        filtered_resp["tags"] = list(target_tags)
        filtered_resp = [filtered_resp] # TODO: change this

    return filtered_resp


# Backend APIs
# TODO: Maybe put these functions in a separate file
def api_search(request):
    if request.method != 'GET':
        err_resp = {
            "message": "The /search API only supports GET"
        }
        return JsonResponse(err_resp, status=405)

    # TODO: Remove stop words so cached queries more likely to match
    query_text = request.GET.get('q', "")
    tags_str = request.GET.get('tags', "")

    if tags_str == "":
        tags = []
    else:
        # TODOS: make tags work
        tags = tags_str.split(",")
        #tags = []
    #print('api_search_here')
    
    print(tags)
    if query_text == "":
        top_cache = SearchCache.objects.order_by("-query_time")[:3]
        #print(top_cache)
        #print(top_cache[0].query_response)
        top_resp = [json.loads(c.query_response) for c in top_cache]
        #print("Empty string Tags is", top_resp[0]['tags'])
        return JsonResponse(top_resp, safe=False)

    else:
        cache_q = SearchCache.objects.filter(query_text=query_text)
        if len(cache_q) > 0:
            cached_resp = json.loads(cache_q[0].query_response)
            filtered_response = _apply_tags(full_response=cached_resp,
                                            target_tags=tags)
            print("Cached tags is: ", filtered_response[0]['tags'])
            return JsonResponse(filtered_response, status=200, safe=False)

        google_search_client = CustomSearchClient(key=config["custom_search_api_key"], cx=config["custom_search_engine_id"])
        
        google_results = google_search_client.query(query_text)
#        print(google_results)
        #google_results2 =google_search_client.query(query_text, start=11)
        #print(google_results)
        #print(google_results2)
        _urls = get_urls_from_search_results(google_results)
        
        #_urls = _urls + get_urls_from_search_results(google_results2)
#        print(_urls)
        #print(_urls2)
        _dates_dic = get_dates_from_search_results(google_results)
        #_dates_dic2 = get_dates_from_search_results(google_results2)
        #_dates_dic.update(_dates_dic2)

        # eval_data, eval_data_dic = get_eval_data(google_results)

        # snippets_dic = get_snippets_from_search_results(google_results)
        #snippets_dic2 = get_snippets_from_search_results(google_results2)
        #snippets_dic.update(snippets_dic2)
        
        articles = {url: parse_article(url) for url in _urls}
        
        
        articles, _urls = filter_articles_by_length(articles, _urls)

        # eval_data = [each for each in eval_data if each['url'] in _urls]

        print("Number of articles selected: ", len(_urls))
        print(len(_urls))
        print(_urls)
        
        #relevant_sentences = get_relevant_sentences(query_text, _urls, articles)
        
        perspective_lst, perspective_dic = get_perspectives(articles, query=query_text)
        
#        articles, _urls, perspective_lst = filter_articles_by_perspectives_relevance(articles, _urls, perspective_lst, query_text)
#        print(len(_urls))
        
        #print(perspective_lst)
        
        equivalent_clusters, url_batches, closest_to_centroids = get_equivalent_clusters_from_titles(query_text, perspective_lst, _urls)
        
        if "how" in query_text.lower()[:6] or "what" in query_text.lower()[:6]:
            center_perspectives = [perspective_lst[i] for i in closest_to_centroids]
        else:
            url_batches, center_perspectives = get_equivalent_clusters_in_stance(query_text, perspective_lst, _urls)
            print("URL batches: ", url_batches)
            
            center_perspectives = [perspective_lst[i] if i!=-1 else [''] for i in center_perspectives ]
        
    
        perspectives = []
        
        stance_dic= {0:"Yes, ", 1: "No, ", 2:"Maybe, "}

        # pair-wise eval
        # csv_dir = "csv_files_eval/"
        # row = ['URL', "Title", "Perspective", "Snippet", "Key argument", "Arguments"]
        # with open(csv_dir+query_text.replace(' ', '_', len(query_text)) +'.csv', 'w') as file:
        #    csvwriter= csv.writer(file)
        #    csvwriter.writerow(row)
        
        # dic_eval ={}
        # dic_eval['query'] = query_text
        # dic_eval['perspectives'] = {}

        for ind, url_batch in enumerate(url_batches):
            current_perspective = {
                "sources": [],
                "articles": [],
                "description": ""
            }
            # the description now is the full texts of arguments
            if url_batch==[]:
                continue
            
            
            if "how" in query_text.lower()[:6] or "what" in query_text.lower()[:6]:
                current_perspective['main_perspective'] = center_perspectives[ind][0] + ", and ..."
            else:
                str_persp = center_perspectives[ind][0]
                str_persp = str_persp.replace("No, ", "")
                str_persp = str_persp.replace("Yes, ", "")
                current_perspective['main_perspective'] = stance_dic[ind] + str_persp  + ", and ..."
            
            # dic_eval['perspectives'][current_perspective['main_perspective']]= {}
            
            date_range = []
            for url in url_batch:
                # Here we parse each HTML page into Article class, with cleaner text + metadata we want
                article = articles[url]
                #sentences_lst = relevant_sentences[url]
                #relevant_sentences_to_sum = [sentences_lst[i][0] for i in range(len(sentences_lst))]
                paragraphs = [p for p in article.text.splitlines() if p]
                
                #print(url)
                # We call each classifier to assign categorical metadata to the articles
                article.dimensions["doc_type"] = classify_document_type(article)
                article.dimensions["topic"] = classify_document_topic(article)
                arguments =get_arguments(article, perspective_dic[url][0])

                arguments, key_argument = get_key_arguments(perspective_dic[url][0], arguments)
                # dic_eval['perspectives'][current_perspective['main_perspective']][key_argument] = {"image": eval_data_dic[url]["image"], "date": eval_data_dic[url]["date"], "org":eval_data_dic[url]["org"], "argument_lst": [arguments[i][0] for i in range(len(arguments))]}
                #print(arguments)
                arguments = '\n'.join([arguments[i][0] for i in range(len(arguments))])
                current_perspective["description"] += arguments + "\n"
                #print(article.title)
                #print(arguments)
                #print()
                
                
    
            
                #article.main_perspective, score, desc = get_main_perspective_and_description(relevant_sentences[url], article)
                article.main_perspective = key_argument #perspective_dic[url][0]
                
                
                
                #print(article.main_perspective)


                # pair-wise eval
                # row = [url, article.title, current_perspective['main_perspective'], snippets_dic[url], key_argument, arguments]
                # with open(csv_dir+query_text.replace(' ', '_', len(query_text)) +'.csv', 'a') as file:
                #    csvwriter= csv.writer(file)
                #    csvwriter.writerow(row)
                
                
                
                #if "main_perspective" not in current_perspective:
                #    max_score = score
                #    current_perspective["main_perspective"] = article.main_perspective
                #    current_perspective["description"] = desc
                #elif score > max_score:
                #    current_perspective["main_perspective"] = article.main_perspective
                #    current_perspective["description"] = desc
                
                if _dates_dic[article.url][:3] in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                    _date = _dates_dic[article.url]
                else:
                    today = datetime.date.today()
                    day_offset= datetime.timedelta(days=[int(s) for s in _dates_dic[article.url].split() if s.isdigit()][0])
                    _date = (today - day_offset).strftime("%B %d, %Y")
                    #print("date", _dates_dic[article.url])
                    #print(article.title)
               
                #print("DATE", _date)
                #print(_dates_dic[article.url])
                #print(article.publish_date)
                date_range.append(_date)
                
                
                
                current_article = {
                    "article_topics": article.dimensions["topic"],
                    "article_type": article.dimensions["doc_type"],
                    "top_img_url": article.top_img_url,
                    "description": arguments,
                    "perspective": article.main_perspective,
                    "source": article.source_name,
                    "title": article.title,
                    "publish_date": _date,
                    "url": article.url,
                    "authors": article.authors,
                    "relates_to_articles": get_related_articles(equivalent_clusters[url], articles)
                }

                current_perspective["sources"].append(article.source_name)
                current_perspective["articles"].append(current_article)
            current_perspective["date_range"] = get_date_range(date_range)
            #print(get_date_range(date_range))
                
            perspectives.append(current_perspective)
            current_perspective["sources"] = list(set(current_perspective["sources"]))

        # pickle_dir = "eval_pickle_files/"
        # pickle_file_name = query_text.replace(' ', '_', len(query_text))
        # pickle_file_name = pickle_file_name.replace('?', '')
        #
        # with open(pickle_dir+ pickle_file_name + "_google" +".pickle", 'wb') as file:
        #     pickle.dump(eval_data, file)
        #
        # with open(pickle_dir+ pickle_file_name + "_perspective" +".pickle", 'wb') as file:
        #     pickle.dump(dic_eval, file)

        title_query_text = format_query(query_text)
        full_response = {
            "title": title_query_text,
            "tags": tags,
            "perspectives": perspectives
        }

        cache_str = json.dumps(full_response, default=str)

        SearchCache.objects.create(query_text=query_text,
                                   query_time=datetime.datetime.now(),
                                   query_response=cache_str)

        filtered_response = _apply_tags(full_response=full_response,
                                        target_tags=tags)
        torch.cuda.empty_cache()
        #print("search tags is: ", filtered_response[0]['tags'])
        return JsonResponse(filtered_response, safe=False)



def api_get_all_tags(request):
    all_tags = []
    for tag in TAG_SET["topic"]:
        all_tags.append({
            "name": tag
        })
#    for tag in TAG_SET["doc_type"]:
#        all_tags.append({
#            "name": tag
#        })

    return JsonResponse(all_tags, status=200, safe=False)



