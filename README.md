# Dev Repo for covid19 application

Sihao's dev instance of the website: [http://macniece.seas.upenn.edu:4020/](http://macniece.seas.upenn.edu:4020/)

Link to our [work document](http://https://docs.google.com/document/d/1wGHFs_TkOwENbyjJNhoj_BwiLMc-1GsvV9NE_tO0880/edit#heading=h.c77wzclj3c3d "work document").

### Instructions on setting up dev environment
The app is built upon [Django](https://www.djangoproject.com/start/ "Django") framework. Here is a [tutorial](https://docs.djangoproject.com/en/3.0/intro/install/ "tutorial") on how you would install django.

Once you have installed django, and every other packages in `requirements.txt`, here is how you would run the server. 

```
$ python manage.py runserver
```
This command will start a server on localhost. If you want it to be public facing -- Replace `<port-number>` with, well, port number. 
```
$ python manage.py runserver 0.0.0.0:<port-number>
```
For the server to function, you will need to provide credentials for a Google Custom Search Engine instance. The next section will show you how to set it up. 
### Steps for setting up Google Custom Search API

1. Follow this [tutorial](https://developers.google.com/custom-search/v1/introduction) to create a Google Custom Search Engine (CSE) instance. Be sure to save the **api key** of your CSE, as it will only be shown to you once during engine creation. 
2. In the [CSE control panel](https://cse.google.com/cse/all), select the list of urls to search for this engine (e.g. nytimes.com, cdc.gov).
3. Again in the [CSE control panel](https://cse.google.com/cse/all), there is a **Search Engine ID**. Save it somewhere. 
4. if you haven't done so, install python package `google-api-python-client`
5. In project directory, there is a config file at `covid19/config/config.json`, in which you will fill the placeholder for **Search Engine ID** and **api key**.

### Steps for setting up Quote Annotator server
1. wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
2. unzip stanford-corenlp-full-2018-10-05.zip
3. cd stanford-corenlp-full-2018-10-05
4. java -Xmx10g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,ner,depparse,coref,quote" -port 9000 -timeout 30000

   **Reference**: https://stanfordnlp.github.io/CoreNLP/quote.html

### Steps for setting up perspectives models
1. Download relevance and stance Roberta models from [google drive](https://drive.google.com/drive/folders/1B0XAWxn7xOsn1bRYCbZcSzh2HiABkx6p)
2. Unzip, and place respective folders in directory model/perspectives/
3. Use command generated [here](https://pytorch.org/get-started/locally/) to download Torch