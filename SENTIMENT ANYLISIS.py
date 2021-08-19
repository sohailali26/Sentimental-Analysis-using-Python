import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

#twitter Api credentials
consumerKey = "xxxxx-xxxxxxx--xxxxxxx--xxxx"
consumerSecret = 'xxxxx--xxxxxxxxxxx---------xxxxxxxx'
accessToken = 'xxxxxxxxxxxx---xxx--xxxxxxxxx--------xxxxxxxx'
accessTokenSecret = 'xxxxxxxxxxxxxxxx---xxxxxxxxxxxxx'

#Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

#set the authentication ojject
authenticate.set_access_token(accessToken, accessTokenSecret) 

#Create the API object while passing in the auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True)

#Extract 100 tweets from the twitter user
posts = api.user_timeline(screen_name = "BillGates", count =100, lang = 'en', tweet_mode="extended")
    
#create a dataframe with a coulmn called Tweets
df = pd.DataFrame( [tweet.full_text for tweet in posts] , columns=['Tweets'])

#Show the first 5 rows of data
df.head()

#Clean the text

#Create a function to clean text
def cleantxt(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text) #Removed @mentions
    text = re.sub(r'#', '', text)   #Removing the '#' symbol
    text = re.sub(r'RT[\s]+', '', text) #Removing RT
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text

#Cleaning the text
df['Tweets']=df['Tweets'].apply(cleantxt)

#show the cleaned text
df

#Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.subjectivity

#Create two new columns
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

#Show the new dataframe with the new columns
df

#Plot the word cloud
allWords = ' '.join( [twts for twts in df['Tweets']] )
cloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119).generate(allWords)

plt.imshow(cloud, interpolation = "bilinear")
plt.axis('off')
plt.show()

#create a function to compute the negatve, neutral and positive
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)

#show the dataframe
df

#Print all of the positive tweets
j=1
sortedDF = df.sort_values(by=['Polarity'])
for i in range(0,sortedDF.shape[0]):
    if(sortedDF['Analysis'][i] == 'Positive'):
        print(str(j) + ') '+sortedDF['Tweets'][i])
        print()
        j = j+1

#print the negative tweets
j=1
sortedDF = df.sort_values(by=['Polarity'], ascending = 'False')
for i in range(0, sortedDF.shape[0]):
    if( sortedDF['Analysis'][i] == 'Negative'):
        print(str(j) +') '+ sortedDF['Tweets'][i])
        print()
        j = j+1

#plot the polarity and subjectivity
plt.figure(figsize=(8,6))
for i in range(0,df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Blue')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

#Get the percentage of positive tweets
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweets']

round( (ptweets.shape[0] / df.shape[0]) *100 , 1)

#Get the percentage of negative tweets
ntweets = df[df.Analysis == 'Negative']
ntweets = ntweets['Tweets']
round( (ntweets.shape[0] / df.shape[0]*100),1)

#show the value counts
df['Analysis'].value_counts()
#plot and visualize the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='bar')
plt.show()