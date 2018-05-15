# tutorial from http://www.pitt.edu/~naraehan/presentation/Movie+Reviews+sentiment+analysis+with+Scikit-Learn.html
# and harish choudhary youtube for web scraping
# and https://gist.github.com/orvi2014/9524db2f5e045b7213d7aa3e8339b13a

import sklearn
from sklearn.datasets import load_files
import re

moviedir = r'C:\Users\Sejal Taneja\Desktop\movie_reviews'

movie_train = load_files(moviedir, shuffle=True)
print(len(movie_train.data))
print(movie_train.target[:])

from sklearn.feature_extraction.text import CountVectorizer
import nltk

sents = ['A rose is a rose is a rose is a rose.',
         'Oh, what a fine day it is.',
         "It ain't over till it's over, I tell you!!"]

foovec = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)

sents_counts = foovec.fit_transform(sents)
print(foovec.vocabulary_)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
sents_tfidf = tfidf_transformer.fit_transform(sents_counts)

movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
movie_counts = movie_vec.fit_transform(movie_train.data)

tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(
    movie_tfidf, movie_train.target, test_size = 0.20, random_state = 12)

clf = MultinomialNB().fit(docs_train, y_train)
y_pred = clf.predict(docs_test)
print(y_pred)


'''
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shined through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']
reviews_new_counts = movie_vec.transform(reviews_new)
reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)

pred = clf.predict(reviews_new_tfidf)

for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie_train.target_names[category]))
'''

import requests
from textwrap import wrap
from bs4 import BeautifulSoup as bsoup

url = 'https://www.imdb.com/title/tt5956100/reviews?ref_=tt_urv'
print(url)

res = requests.get(url)

print("\nNEG OR POS BASED ON REVIEW TITLES\n")

mylist=[]
soup = bsoup(res.text, "html.parser")
for my_tag in soup.find_all(class_="title"):
    message_text = my_tag.getText()
    mylist.append(message_text)

mylist_counts = movie_vec.transform(mylist)
mylist_tfidf = tfidf_transformer.transform(mylist_counts)

pred = clf.predict(mylist_tfidf)

for review, category in zip(mylist, pred):
    print('%r => %s' % (review, movie_train.target_names[category]))

print("\nNEG OR POS BASED ON RATINGS\n")

for my_tag in soup.find_all(class_='rating-other-user-rating'):
    message_text = my_tag.getText()
    message_text = re.sub(r"\n", '', message_text)
    if int(message_text[0]) < 5:
        print("Rating: ", message_text, "==> neg", )
    elif int(message_text[0]) > 5:
        print("Rating: ", message_text, "==> pos", )

print("\nNEG OR POS BASED ON DETAILED REVIEWS\n")

myreview = []
soup = bsoup(res.text, "html.parser")
for my_tag in soup.find_all(class_="text show-more__control"):
    message_text = my_tag.getText()
    myreview.append(message_text)

myreview_counts = movie_vec.transform(myreview)
myreview_tfidf = tfidf_transformer.transform(myreview_counts)

pred = clf.predict(myreview_tfidf)

for review, category in zip(myreview, pred):
    i = wrap(review, 170)
    for x in range(0, len(i)):
        print(i[x])
    print('=> %s' % (movie_train.target_names[category]))
    print('\n')

