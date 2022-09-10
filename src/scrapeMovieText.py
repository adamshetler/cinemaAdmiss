import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search

data = pd.read_csv("data/out/cinemaFeatures.csv")


for i in range(len(data)):
    movie = data.Title[i]
    query = movie + " film wiki"
    urls = []
    # TODO: make sure wiki page is always selected (even if not first res)
    for search_res in search(query, tld="com", num=10, stop=10, pause=2):
        urls.append(search_res)

    #create page object
    page = requests.get(urls[0])
    soup = BeautifulSoup(page.text, 'lxml')
    # TODO: scrape first paragraph from all wikis
    # TODO: scrape wiki table for numerical features
    # TODO: scrape wiki bullet points for cast
    # headers = ['Premise', 'Plot', 'Cast', 'Actors']
    headers = ['Premise', 'Reception', 'Production', 'Plot', 'Release', 'Cast', 'Filming',
               'Music', 'Soundtrack', 'People appearing in the film', 'Production and release',
               'Critical reception', 'Synopsis', 'Reviews', 'Themes']

    # scrape all paragraphs under select headers on movie's wiki page -  append p's to paragraph_list
    # TODO: fix duplicate paragraphs
    paragraphs_list = []
    for header in headers:
        targets = soup.select('h2', id=header)
        for target in targets:
            for sibling in target.find_next_siblings():
                if sibling.name == "p":
                    paragraphs_list.append(sibling.text)
                else:
                    break
    #remove duplicate paragraphs from paragraphs_list
    final_list = []
    [final_list.append(x) for x in paragraphs_list if x not in final_list]
    paragraphs_list = final_list

    # selector = ', '.join([f'h2:-soup-contains("{header}") + p' for header in headers])
    # paragraphs_list = [i.text for i in soup.select(selector)]

    if len(paragraphs_list) > 0:
        combined_text = ""
        for paragraphs in paragraphs_list:
            combined_text += " " + paragraphs
        print(f'Text info found for {movie}')
        print(combined_text)
        data.loc[i, "scrapedInfo"] = combined_text
        #data.loc[i, "scrapePlot"] = paragraphs_list[0]
    else:
        print(f'No plot text found for movie: {movie}')


data.to_csv("data/out/cinemaFeaturesTextScrape.csv")








