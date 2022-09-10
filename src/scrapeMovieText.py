import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search

data = pd.read_csv("data/out/cinemaFeatures.csv")

for i in range(len(data)):
    movie = data.Title[i]
    query = movie + " film wiki"
    urls = []
    for search_res in search(query, tld="com", num=10, stop=10, pause=2):
        urls.append(search_res)

    #create page object
    page = requests.get(urls[0])
    soup = BeautifulSoup(page.text, 'lxml')

    # headers = ['Premise', 'Plot', 'Cast', 'Actors']
    headers = ['Premise', 'Reception', 'Production', 'Plot', 'Release', 'Cast', 'Filming', 'Music', 'Soundtrack',
               'People appearing in the film', 'Production and release', 'Critical reception', 'Synopsis', 'Reviews', 'Themes']

    selector = ', '.join([f'h2:-soup-contains("{header}") + p' for header in headers])
    paragraphs_list = [i.text for i in soup.select(selector)]
    if len(paragraphs_list) > 0:
        print(f'Plot text found for movie: {movie}')

        combined_text = ""
        for paragraphs in paragraphs_list:
            combined_text += " " + paragraphs
        data.loc[i, "scrapedInfo"] = combined_text
        #data.loc[i, "scrapePlot"] = paragraphs_list[0]
    else:
        print(f'No plot text found for movie: {movie}')


data.to_csv("data/out/cinemaFeaturesTextScrape.csv")








