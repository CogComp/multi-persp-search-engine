from googleapiclient.discovery import build

entry_url = "https://www.googleapis.com/customsearch/v1/siterestrict"


class CustomSearchClient:
    def __init__(self, key, cx):
        """

        :param key: API key
        :param cx: custom search engine id
        :param q: query
        """
        self._cx = cx

        self._service = build("customsearch", "v1", developerKey=key)
        #print("My google search is", self._service)

    def query(self, q, **kwargs):
        # search engine key words: 'covid-19' 'corona virus' 'coronavirus' 'covid19' 'Covid19' 'Covid-19'
        # args: exactTerms='covid-19',
        res = self._service.cse().list(q=q, cx=self._cx, dateRestrict='y2', **kwargs).execute()
        #print(res)
        if 'items' in res:
            return res['items']
        else:
            return []


def get_urls_from_search_results(results):
    return [res['link'] for res in results]

def get_dates_from_search_results(results):
    urls = [res['link'] for res in results]
    dates = [res['snippet'].split(' ...')[0] for res in results]
    dates_dic = {}
    for i, url in enumerate(urls):
        dates_dic[url] = dates[i]
    return dates_dic
    
def get_snippets_from_search_results(results):
    urls = [res['link'] for res in results]
    snippets_dic = {}
    snippets = [res['snippet'] for res in results]
    for i, url in enumerate(urls):
            snippets_dic[url] = snippets[i]
    return snippets_dic

def get_eval_data(results):
    urls = [res['link'] for res in results]
    data = []
    data_dic = {}
    titles = [res['title'] for res in results]
    snippets = [res['snippet'] for res in results]
    #print(snippets)
    date = [each.split('...')[0] for each in snippets]
    snippets = ['...'.join(each.split('...')[1:])+" ..." for each in snippets]
    #print(snippets)
    formattedUrl = [res['formattedUrl'] for res in results]
    display_link = [res['displayLink'] for res in results]
    images_url = [res['pagemap']['cse_image'][0]['src'] for res in results]
    for i, url in enumerate(urls):
        data_dic[url] = {"image": images_url[i], "date": date[i], "org": display_link[i]}
        data.append({"url": urls[i], "title": titles[i], "date": date[i], "snippet": snippets[i], "image": images_url[i], "org": display_link[i]})
    return data, data_dic


if __name__ == '__main__':
    from sys import argv
    if len(argv) != 4:
        print("Usage: python google_custom_search.py [api-key] [cx] [query]")
        exit(1)
    cli = CustomSearchClient(argv[1], argv[2])
    r = cli.query(argv[3])
    print(len(r))
    print(type(r))
    print(type(r[0]))
    print(r[0].keys())
    print(r[0]['pagemap'])
    print([c["title"] for c in r])
    print([c["displayLink"] for c in r])
    #print([c["pagemap"] for c in r])
    #print([c["snippet"] for c in r])
    get_eval_data(r)
    #print(r[0]['pagemap']['metatags'])
    #print(get_dates_from_search_results(r))
    #print(r[0])
