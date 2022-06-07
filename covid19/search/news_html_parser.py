from newspaper import Article, Config
from covid19 import apis
from covid19.ie.extract_info import *
import random

config = Config()
config.memoize_articles = False
config.language = 'en'

# Test quotes: remove later.
quote_dictionary = {"As I've said many times, we look at it from a pure health standpoint. "
                    "We make a recommendation. Often the recommendation is taken. Sometimes it's not. "
                    "But it is what it is. We are where we are right now.": "Dr. Anthony Fauci",
                    "The media chatter is ridiculous — President Trump is not firing Dr. Fauci.": "White House Deputy Press Secretary Hogan Gidley",
                    "This is not time for politics, and it is no time to fight. I put my hand out in total partnership"
                    " and cooperation with the president. If he wants a fight he’s not going to get it from me. Period.":
                    "New York Governor Andrew Cuomo"}


def parse_article(url:str,
                  language: str='en') -> apis.Article:
    """
    TODO: support other languages
    :param url:
    :param language:
    :return:
    """
    a = Article(url, config=config)
    a.download()
    a.parse()
    # links = extract_links(url)
    # people, orgs = extract_people_orgs(a.text)
    # quotes = extract_quotes(a.text)
    quotes = extract_quotes_baseline(a.text)
    article = apis.Article(text=a.text,
                           url=url,
                           publish_date=a.publish_date,
                           top_img_url=a.top_image,
                           authors=a.authors,
                           # links=links,
                           # people=people,
                           # orgs=orgs,
                           quotes=quotes,
                           title=a.title)

    return article
#    return a


if __name__ == '__main__':
    # a = parse_article('https://www.nytimes.com/2014/11/18/upshot/got-milk-might-not-be-doing-you-much-good.html')
    a = parse_article('https://www.reuters.com/article/us-health-coronavirus-france-idUSKCN24H15L')
    #a = parse_article('https://www.bbc.com/news/world-us-canada-52264860')
    print(a.text)
    # print(a.publish_date)
    # print(a.authors)
    # print(a.links)
    # print(a.people)
    # print(a.orgs)
    # print(a.quotes)
    # print([p for p in a.text.splitlines() if p])
