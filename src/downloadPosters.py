from icrawler.builtin import GoogleImageCrawler
import pandas as pd


data = pd.read_csv("data/out/cinemaFeatures.csv")

# download movie posters and save in data/download/posters
# NOTE: each film poster is in a folder with the film title due to the way GoogleImageCrawler saves downloads
for i in range(len(data)):
    movie = data.Title[i].strip()
    movie_file = movie.replace(" ", "_")

    google_Crawler = GoogleImageCrawler(storage = {'root_dir': f'data/download/posters/{movie_file}'})
    google_Crawler.crawl(keyword = f'{movie} film poster', max_num = 1)