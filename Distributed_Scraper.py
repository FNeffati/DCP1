########################################################################################################
import re
import time
import urllib

import pyspark
from bs4 import NavigableString, BeautifulSoup

page_indexes = [1]  # here you should insert all the indexes you want to go up to, one by one
sc = pyspark.SparkContext("local[*]", "Test Context")
rdd = sc.parallelize(page_indexes)


def landing_page(page_index):
    """
    this will make a HTTP request and return the raw html of a landing page
    a landing page is a page that has 20 article previews in it

                    -THIS IS TO BE MAPPED-

    :param page_index: The index to that specific Landing Page
    :return: The RAW HTML FROM THAT PAGE
    """

    url = f'https://archiveofourown.org/tags/Star%20Wars%20-%20All%20Media%20Types/works?commit=Sort+and+Filter' \
          f'&include_work_search%5Brating_ids%5D%5B%5D=10&' \
          f'page=${page_index}&work_search%5Bcomplete%5D=&work_search%5Bcrossover%5D=&work_search%5Bdate_from%5D' \
          f'=&work_search%5Bdate_to%5D=&work_search%5Bexcluded_tag_names%5D=&work_search%5Blanguage_id%5D' \
          f'=&work_search%5Bother_tag_names%5D=&work_search%5Bquery%5D=&work_search%5Bsort_column%5D=revised_at' \
          f'&work_search%5Bwords_from%5D=&work_search%5Bwords_to%5D= '
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_2) AppleWebKit/605.1.15 (KHTML, '
                             'like Gecko) Version/14.0.3 Safari/605.1.15'}

    try:
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print('Too many requests!---SLEEPING---')
            print('we should restart on page', i)
            print('we should restart with this url:', url)
        raise

    return resp


landing_page_rdd = rdd.map(landing_page)


def get_all_articles_from_landing_page(page_raw_HTML):
    """
    find all the "li" elements with the attribute "role" set to "article".

                    -THIS IS NOT TO BE MAPPED.-

    :param page_raw_HTML: The HTML of the landing page ( the page that contains 20 links to articles)
    :return: the links to the specific articles
    """
    bs = BeautifulSoup(page_raw_HTML, 'lxml')
    return bs.find_all('li', {'role': 'article'})


all_articles_on_landing = landing_page_rdd.map(get_all_articles_from_landing_page)


def process_landing_page_article(article):
    """
    Returns the specific info (listed below) that can be extracted from the landing page about that one specific article.

                    -THIS IS NOT TO BE MAPPED.-

    :param article: HTML of that article
    :return:row: A list of lists
    """
    row = []
    article = BeautifulSoup(article, 'lxml')
    title = article.find('h4', {'class': 'heading'}).find('a').text
    try:
        author = article.find('a', {'rel': 'author'}).text

    except:
        author = 'Anonymous'

    identifier = article.find('h4', {'class': 'heading'}).find('a').get('href')[7:]
    date_updated = article.find('p', {'class': 'datetime'}).text
    rating = article.find('span', {'class': re.compile(r'rating\-.*rating')}).text
    pairing = article.find('span', {'class': re.compile(r'category\-.*category')}).text
    warning = article.find('span', {'class': re.compile(r'warning\-.*warnings')}).text
    completeness = article.find('span', {'class': re.compile(r'complete\-.*iswip')}).text
    language = article.find('dd', {'class': 'language'}).text
    count = article.find('dd', {'class': 'words'}).text

    if len(count) > 0:
        word_count = count
    else:
        word_count = '0'
    chapter = article.find('dd', {'class': 'chapters'}).text.split('/')[0]
    try:
        comment = article.find('dd', {'class': 'comments'}).text
    except:
        comment = '0'
    try:
        kudo = article.find('dd', {'class': 'kudos'}).text
    except:
        kudo = '0'
    try:
        bookmark = article.find('dd', {'class': 'bookmarks'}).text
    except:
        bookmark = '0'
    try:
        hit = article.find('dd', {'class': 'hits'}).text
    except:
        hit = '0'

    row.append(
        [title, author, identifier, date_updated, rating, pairing, warning, completeness, language, word_count,
         chapter, comment, kudo, bookmark, hit])

    return row


all_articles_on_landing_data = all_articles_on_landing.map(process_landing_page_article)


def get_article_work_id(article_soup):
    """
    This takes a specific article soup and returns the work ID which will be used for mutliple things later.
    :param article_soup: T
    :return:
    """
    article_soup = BeautifulSoup(article_soup, 'lxml')
    return article_soup.find('h4', {'class': 'heading'}).find('a').get('href')


all_article_work_ids = all_articles_on_landing.map(get_article_work_id)


def get_fiction_soup(work_id):
    url = 'https://archiveofourown.org' + work_id + '?view_adult=true&show_comments=true&view_full_work=true'
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_2) AppleWebKit/605.1.15 (KHTML, '
                             'like Gecko) Version/14.0.3 Safari/605.1.15'}
    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req)
    print('Successfully opened fiction:', url)
    bs = BeautifulSoup(resp, 'lxml')
    time.sleep(5)
    return bs


all_fiction_soups = all_article_work_ids.map(get_fiction_soup)


def get_publish_date(fiction_soup):
    fiction_soup = BeautifulSoup(fiction_soup, 'lxml')
    return fiction_soup.find('dd', {'class': 'published'}).text


all_publish_dates = all_fiction_soups.map(get_publish_date)


def get_fiction_content(fiction_soup):
    fiction_soup = BeautifulSoup(fiction_soup, 'lxml')
    return fiction_soup.find('div', {'id': 'chapters'}).text.strip()


all_fiction_content = all_fiction_soups.map(get_fiction_content)


def get_tags(fiction_soup):
    fiction_soup = BeautifulSoup(fiction_soup, 'lxml')
    tags = []
    for child in fiction_soup.find('ul', {'class': 'tags commas'}).children:
        if isinstance(child, NavigableString):
            pass
        else:
            tags.append(child.text.strip())
    return ', '.join(tags)


all_tags = all_fiction_soups.map(get_tags)


def get_summary(fiction_soup):
    fiction_soup = BeautifulSoup(fiction_soup, 'lxml')
    try:
        out = fiction_soup.find('blockquote', {'class': 'userstuff summary'}).text.strip()
        return out
    except:
        return ''


all_summaries = all_fiction_soups.map(get_summary)


def get_rows(rdd1, rdd2, rdd3, rdd4, rdd5):
    """
    TODO: FIND A WAY TO TAKE MULTIPLE RDDs and CREATE 1 SINGLE ROW OUT OF THEM
    # IDEA: Have all of the RDDs use the work ID as a unique identifier
    # And keep reducing by key 1 column at a time until I get all of the columns
    :return:
    """
    combined_rdd = rdd1.zip(rdd2).zip(rdd3).zip(rdd4).zip(rdd5).map(
        lambda x: [x[0][0][0], x[0][0][1], x[0][1], x[1], x[2]])
    result = combined_rdd.collect()

    return result



