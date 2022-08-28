import settings
import tweepy
import dataset
from textblob import TextBlob
from datafreeze.app import freeze

db = dataset.connect(settings.CONNECTION_STRING)

result = db[settings.TABLE_NAME].all()
#this is the step I am struggling with
dataset.freeze(result, format='csv', filename=settings.CSV_NAME)