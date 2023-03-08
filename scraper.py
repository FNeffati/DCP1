import time
import csv
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import NavigableString
import re
import urllib.request

'''
NOTE: Need to have a SomeContext folder in the same directory to dump files into

Begins by getting the "content" of all pages in our search. Should be ~3000 for general star wars fanfiction
Each work in the "content" file is scraped for its various metrics, appended to SomeName.csv
As items are populated into SomeName they're also populated into SomeText (with story body, summary, and more)  
Currently configured to scrape 1 search page / 20 works
'''

headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'
}


def process_basic(page_content):
    bs = BeautifulSoup(page_content, 'lxml')
    titles = []
    authors = []
    ids = []
    date_updated = []
    ratings = []
    pairings = []
    warnings = []
    complete = []
    languages = []
    word_count = []
    chapters = []
    comments = []
    kudos = []
    bookmarks = []
    hits = []

    for article in bs.find_all('li', {'role': 'article'}):
        titles.append(article.find('h4', {'class': 'heading'}).find('a').text)
        try:
            authors.append(article.find('a', {'rel': 'author'}).text)
        except:
            authors.append('Anonymous')
        ids.append(article.find('h4', {'class': 'heading'}).find('a').get('href')[7:])
        date_updated.append(article.find('p', {'class': 'datetime'}).text)
        ratings.append(article.find('span', {'class': re.compile(r'rating\-.*rating')}).text)
        pairings.append(article.find('span', {'class': re.compile(r'category\-.*category')}).text)
        warnings.append(article.find('span', {'class': re.compile(r'warning\-.*warnings')}).text)
        complete.append(article.find('span', {'class': re.compile(r'complete\-.*iswip')}).text)
        languages.append(article.find('dd', {'class': 'language'}).text)
        count = article.find('dd', {'class': 'words'}).text
        if len(count) > 0:
            word_count.append(count)
        else:
            word_count.append('0')
        chapters.append(article.find('dd', {'class': 'chapters'}).text.split('/')[0])
        try:
            comments.append(article.find('dd', {'class': 'comments'}).text)
        except:
            comments.append('0')
        try:
            kudos.append(article.find('dd', {'class': 'kudos'}).text)
        except:
            kudos.append('0')
        try:
            bookmarks.append(article.find('dd', {'class': 'bookmarks'}).text)
        except:
            bookmarks.append('0')
        try:
            hits.append(article.find('dd', {'class': 'hits'}).text)
        except:
            hits.append('0')

    df = pd.DataFrame(list(
        zip(titles, authors, ids, date_updated, ratings, pairings, warnings, complete, languages, word_count, chapters,
            comments, kudos, bookmarks, hits)))

    print(df)
    print('Successfully processed', len(df), 'rows!')

    with open('SomeName.csv', 'a', encoding='utf8') as f:
        df.to_csv(f, header=False, index=False)
    temp = pd.read_csv('SomeName.csv')
    print('Now we have a total of', len(temp), 'rows of data!')
    print('================================')


def process_articles(articles, start_index=0, start_index2=1):
    for i, article in enumerate(articles[start_index:]):
        print('* We are working on article', i + start_index)
        work_id = article.find('h4', {'class': 'heading'}).find('a').get('href')
        try:
            row = article_to_row(work_id=work_id, article=article, headers=headers, start_index=start_index2)
            with open('SomeText.csv', 'a', encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print('---Too many requests when accessing ARTICLE---')
                print('We should try this ID later:', work_id, 'which has an index of', i)
                break
            raise


def article_to_row(work_id, article, headers, start_index=1):
    bs = open_fic(work_id, headers=headers)
    publish_date = bs.find('dd', {'class': 'published'}).text
    content = bs.find('div', {'id': 'chapters'}).text.strip()
    return [work_id[7:], get_tags(article), get_summary(article), publish_date, content]


def get_tags(article):
    tags = []
    for child in article.find('ul', {'class': 'tags commas'}).children:
        if isinstance(child, NavigableString):
            pass
        else:
            tags.append(child.text.strip())
    return ', '.join(tags)


def get_summary(article):
    try:
        out = article.find('blockquote', {'class': 'userstuff summary'}).text.strip()
        return out
    except:
        return ''


def getContent(url, start_page=1, end_page=1):
    basic_url = url
    # should be of the form: "https://archiveofourown.org/tags/###TAG###/works?page="

    for i in range(start_page, end_page + 1):
        url = basic_url + str(i)
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            pageName = "./SomeContent/" + str(i) + ".html"
            with open(pageName, 'w') as f:
                f.write(resp.read().decode('utf-8'))
                print(pageName, end=" ")
            time.sleep(5)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print('Too many requests!---SLEEPING---')
                print('we should restart on page', i)
                print('we should restart with this url:', url)
                break
            raise


def open_fic(work_id, headers):
    url = 'https://archiveofourown.org' + work_id + '?view_adult=true&show_comments=true&view_full_work=true'
    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req)
    print('Successfully opened fiction:', url)
    bs = BeautifulSoup(resp, 'lxml')
    time.sleep(5)
    return bs


url = "https://archiveofourown.org/tags/Star%20Wars%20-%20All%20Media%20Types/works?commit=Sort+and+Filter&include_work_search%5Brating_ids%5D%5B%5D=10&work_search%5Bcomplete%5D=&work_search%5Bcrossover%5D=&work_search%5Bdate_from%5D=&work_search%5Bdate_to%5D=&work_search%5Bexcluded_tag_names%5D=&work_search%5Blanguage_id%5D=&work_search%5Bother_tag_names%5D=&work_search%5Bquery%5D=&work_search%5Bsort_column%5D=revised_at&work_search%5Bwords_from%5D=&work_search%5Bwords_to%5D=&page="

getContent(url, start_page=1, end_page=1)

# Create a file to dump information into
header = ['Title', 'Author', 'ID', 'Date_updated', 'Rating', 'Pairing', 'Warning', 'Complete', 'Language', 'Word_count',
          'Num_chapters', 'Num_comments', 'Num_kudos', 'Num_bookmarks', 'Num_hits']
with open('SomeName.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    # writer.writerow(header)


header2 = ['Work ID', 'Tags', 'Summary', 'Publish Date', 'Body']
with open('SomeText.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f)
    writer.writerow(header2)
    # writer.writerow(header)

# Once all the search pages are saved in SomeContent, run this
totalPages = 1
ix = 0
for i in range(1, totalPages + 1):
    pageName = "./SomeContent/" + str(i) + ".html"
    with open(pageName, mode='r', encoding='utf8') as f:
        print('========We are opening page', i, '========')
        page = f.read()
        process_basic(page)  # get basic metrics from search page
        bs = BeautifulSoup(page, 'lxml')
        l_articles_on_page = bs.find_all('li', {'role': 'article'})
        if ix != 0:
            process_articles(articles=l_articles_on_page, start_index=ix, start_index2=1)
            ix = 0
        else:
            process_articles(articles=l_articles_on_page, start_index=ix, start_index2=1)
        temp = pd.read_csv('SomeText.csv')
        print('Now we have a total of', len(temp), 'rows of data!')
