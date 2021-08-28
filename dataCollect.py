'''
Code of data collection part:
Call twitter api to collect the comments into a csv file

Input: 
1. In TweepyApi() function, change the CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_SECRET to your owns
2. In Main function, change the twitterAccountName list to the cadidates twitter account names
3. In Main function, change the Names list(final csv file name) as the corresponding twitterAccountName list

Compile: 
Run 'python dataCollect.py'

Output: 
You will see two comment csv files of each cadidate in current directory 
'''
import csv
import tweepy

'''
Function TweepyApi():
With keys of CONSUMER and OAUTH to get the twitter api access

Input: 
None

Output:
Twitter api access
'''
def TweepyApi():
    CONSUMER_KEY = 'AIJCTAGDnloQv6Vv6nXINeU6V'
    CONSUMER_SECRET = 'oZhGt3o5RUt1Guf3Q4UM5VE9JwpK6g6staOdFtdBtP2nvQV559'
    OAUTH_TOKEN = '1235013351547318277-xUgk3Ef69gMihGYw2TRZf5tfGRRuv8'
    OAUTH_TOKEN_SECRET = 'IJ67eek7IYppxCuRcSVadvvSZ4cEohlSzYYE9Mfz6LRTq'

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

def GetCsvData(Api, TwitterAccountName, CsvFile):
    replies = []
    for tweet in tweepy.Cursor(Api.search, q='to:'+TwitterAccountName, result_type='recent', timeout=999999).items(1000):
        if hasattr(tweet, 'in_reply_to_status_id_str'):
            replies.append(tweet)

    with open(CsvFile, 'a+', encoding='utf-8') as f:
        csv_writer = csv.DictWriter(f, fieldnames=('user', 'text'))
        csv_writer.writeheader()
        for tweet in replies:
            row = {'user': tweet.user.screen_name, 'text':tweet.text.replace('\n', ' ')}
            csv_writer.writerow(row)

'''
Function Main():
Start the process of data collection

Input:
Change the Names and twitterAccountName to correspoding twitter account name and name

Output:
Return two comment csv files of candidates

'''
if __name__=='__main__':
    Api = TweepyApi()
    Names = ['Harris', 'Pence']
    twitterAccountName = ['KamalaHarris', 'Mike_Pence']
    
    for i in range(2):
        GetCsvData(Api, twitterAccountName[i], '{}_data.csv'.format(Names[i]))
