import re
import json
from typing import List, Dict
from urllib import request

from bs4 import BeautifulSoup

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
import en_core_web_sm
from spacy.matcher import Matcher

class StanfordNLP:
    """
    Wrapper Class for Stanford Core NLP server
    """

    def __init__(self,
                 host: str,
                 port: int,
                 timeout: int=30000):

        # self.nlp = StanfordCoreNLP(host, port=port, timeout=timeout)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,depparse,coref,quote',
            'pipelineLanguage': 'en',
            'outputFormat': 'json',
            "quote.attributeQuotes": "false",
            "coref.algorithm": "statistical",
        }

    # def annotate(self,
    #              sentence: str) -> Dict:
    #
    #     results = self.nlp.annotate(sentence, properties=self.props)
    #     if results.startswith("java"): # Error happened
    #         print(results)
    #         return {}
    #     else:
    #         return json.loads(self.nlp.annotate(sentence, properties=self.props))


nlp = en_core_web_sm.load()
porter_stemmer = PorterStemmer()
# sNLP = StanfordNLP(host="http://macniece.seas.upenn.edu", port=4021)  # TODO: move this to config file


def extract_links(url: str) -> List[str]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
    req = request.Request(url=url, headers=headers)
    html_page = request.urlopen(req).read()
    soup = BeautifulSoup(html_page, features="html.parser")
    links = []
    for link in soup.findAll('a', attrs={'href': re.compile("(^http://|^https://)")}):
        links.append(link.get('href'))
    return links


def extract_people_orgs(text: str):
    people, orgs = list(), list()
    spacy_art = nlp(text)
    for x in spacy_art.ents:
        if x.label_ == 'PERSON':
            people.append(x.text)
        if x.label_ == 'ORG':
            orgs.append(x.text)
    return list(set(people)), list(set(orgs))


# def extract_quotes(text: str) -> List[tuple]:
#     """
#
#     :param text:
#     :return: List of tuples, each tuple consists of (quote, speaker)
#     """
#     # _res = sNLP.annotate(text)
#     print(_res)
#     if 'quotes' not in _res:
#         return []
#
#     quotes = _res['quotes']
#     quotes = [(_dict["text"], _dict["canonicalSpeaker"]) for _dict in quotes]
#     # TODO: Filter quotes that we actually want
#     # Filter quotes that are too short and with unknown author
#     quotes = filter_quotes(quotes)
#     return quotes


REPORTING_VERBS = ['say', 'said', 'tell', 'insist', 'apologize', 'ask', 'advise', 'agree', 'explain',
                       'decide', 'encourage', 'promise', 'recommend', 'remind', 'suggest', 'warn']


def filter_quotes(quotes: List[tuple]) -> List[tuple]:
    return [(_text, _author) for _text, _author in quotes if _author != "Unknown" and len(_text) > 50]


def extract_quotes_baseline(text: str) -> List[tuple]:

    quoted_sents = list()
    sents = sent_tokenize(text)
    for sent in sents:
        proc_sent = nlp(sent)

        # Get nearest PERSON as speaker, in any
        speaker = "Unknown"
        for ent in proc_sent.ents:
            if ent.label_ == "PERSON":
                speaker = ent.text

        # First check if there is a direct quote in the sentence
        matcher = Matcher(nlp.vocab)
        matcher.add('DIRECT_QUOTE', None, [{'ORTH': '“'}, {'OP': '*'}, {'ORTH': '”'}])

        matches = matcher(proc_sent)
        if matches:
            for match_id, start, end in matches:
                sent = " ".join([_t.text for _t in proc_sent[start + 1:end]])
                quoted_sents.append((sent, speaker))

        else:
            # Next if there's no direct quotes, look for indirect quotes
            tok_pos = 0

            for tok in proc_sent:
                tok_lem = tok.lemma_
                if tok.dep_ == 'ROOT' and tok_lem in REPORTING_VERBS:
                    sent = " ".join([_t.text for _t in proc_sent[tok_pos+1:]])

                    quoted_sents.append((sent, speaker))
                    break

                tok_pos += 1

    quoted_sents = filter_quotes(quoted_sents)
    return quoted_sents


if __name__ == '__main__':
    import time
    start_time = time.perf_counter()
    url = 'President Trump on Sunday said the administration was preparing to use the Defense Production Act to compel an unspecified U.S. facility to increase production of test swabs by over 20 million per month. The announcement came during his Sunday evening news conference, after he defended his response to the pandemic amid criticism from governors across the country claiming that there has been an insufficient amount of testing to justify reopening the economy any time soon. “We are calling in the Defense Production Act,” Mr. Trump said. He added, “You’ll have so many swabs you won’t know what to do with them.” He provided no details about what company he was referring to, or when the administration would invoke the act. And his aides did not immediately respond when asked to provide more details. “We already have millions coming in,” he said. “He added, “In all fairness, governors could get them themselves. But we are going to do it. We’ll work with the governors and if they can’t do it we’ll do it.” Public health experts have said testing would need to at least double or even triple to justify even a partial reopening of the country’s economy, and business leaders reiterated that message in a conference call with Mr. Trump last week.'
    links = extract_quotes_baseline(url)
    print(links)
    print("Elasped Time: {} s", time.perf_counter() - start_time)
