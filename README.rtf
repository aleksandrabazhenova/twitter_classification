# TWITTER PROJECT
## PART I: VIRAL TWEETS CLASSIFICATION USING A SUPERVISED MACHINE LEARNING ALGORITHM

### INTRODUCTION

This project uses a K-Nearest Neighbor classification algorithm to predict whether a tweet will go viral.
Initially, a broad definition of a 'viral' tweet is established as one which has been 'shared more times than 10 times the user's followers count'.
The reader should note that this definition is arbitrary.

### SCOPE

#### GOALS

1. Use a K-Nearest Neighbor classifier to predict whether a tweet will go viral to 95% accuracy
2. Determine which features of a tweet are most likely to cause a tweet to go viral:
	i. number of hashtags in the tweet;
	ii. number of words or characters;
	iii. number of links;
	iv. number of followers the user has;
	v. specific language used;
	vi. number of user mentions.

#### ACTIONS

1. Using the features that have the strongest correlation to whether a tweet will go viral, write a tweet, which the K-Nearest Neighbor classifier
will predict to go viral.
2. Post the tweet to see whether it will, indeed, go viral.

#### DATA

*1. Where is the data sourced from?*<br>

- The data was sourced from Twitter and accessed through Codecademy's Data Science Career Path.

*2. How is the data stored?*<br>

- The data is store in raw format as a single JSON file.

*3. What are the columns in the data?*<br>

- There are 31 columns in the data; they are as follows:

['created_at', 'id', 'id_str', 'text', 'truncated', 'entities', 'metadata', 'source', 'in_reply_to_status_id',
'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name',
'user', 'geo', 'coordinates', 'place', 'contributors', 'retweeted_status', 'is_quote_status', 'retweet_count',
'favorite_count', 'favorited', 'retweeted', 'lang', 'possibly_sensitive', 'quoted_status_id', 'quoted_status_id_str',
'extended_entities', 'quoted_status', 'withheld_in_countries']

*4. How many entries are there in total?*<br>

- There are 11,099 entrie

*5. Is there missing information?*<br>

- The data doesn't contain any missing information.
- Those columns, which contain features that a tweet doesn\'t necessarily have to have, do contain NaN cells.
For example, if a tweet is not a retweet of someone else\'92s tweet the retweeted_status column cell would contain a NaN.

6. Level of granularity of the data?*<br>

- The data is very granular, and contains a lot of detail about each tweet, the tweet\'92s user, and those users, whose tweets were retweeted.

*7. How far back does the data go?*<br>

- The data is a 28 second snapshot of tweets created between 2018-07-31 13:34:13 and 2018-07-31 13:34:40 UTS.
There are 28 unique date-times for the 11,099 tweets.

*8. How often is the data collected?*<br>

- The data was collected each second between 2018-07-31 13:34:13 and 2018-07-31 13:34:40 UTS.

*9. How often does new data come in?*<br>

- No new data is added to the dataframe.

*10. Does new data overwrite old fields or does is add new rows?*<br>

- NA.

*11. Is there a collection bias in the data?*<br>

- It is likely that a collection bias exists, given that all 11,099 tweets are from the same 28 seconds in the same timezone.
This is because world events and news reports can heavily influence the content of world-wide twitter conversations at any one time.
- There are only 17 row entries, which contain geolocation information. For those 17 entries, the following coordinates are stated:

[20.21, 102.62]
[51.40966808, -0.30390349]
[44.26928641, -76.51193182]
[45.2029, -93.387]
[64.1333, -21.9333]
[35.7795897, -78.6381787]
[39.90854909, 116.47548753]
[50.82111066, -0.14220751]
[42.5459227, -71.9106308]
[19.2, -96.1333]
[26.3582748, -81.7458845]
[37.60024833, -0.80567856]
[52.518391, 13.401251]
[9.0, 8.0]
[43.7166, -79.3407]
[33.7489954, -84.3879824]
[37.80559334, -122.41312804]

- This small sample has users from all over the world, indicating that there is unlikely to be any location bias in the rest of the data despite
absence of location data.

#### ANALYSIS

1. Distribution of retweet count for each tweet
2. Analysis and creation of additional features
	2.1. Correlation between followers/ friends count and retweeted count
	2.2 Bar chart of average retweet count per number of hashtags
	2.3 Bar chart of average retweet count per number of links
	2.4 Bar chart of average retweet count per number of user mentions
3. KNN classification model analysis with the above features
4. Analysis of specific language used with natural language processing
5. Correlation between specific language and retweeted count
6. Design of viral tweet

### DISCUSSION AND EVALUATION

#### <u> Definition of 'Viral'</u>

- The maximum retweet count in the dataset is 413,719 retweets.

- The average retweet count in the dataset is 2,777.96 retweets.

- The median retweet count is 13.

- Number of tweets with retweet count over 100,000 is 24.

- Figure 1.1 shows a KDE plot of retweet count, with a KDE spike almost at zero and some much smaller KDE remnants over the rest of the retweet count value, 
stretching over 400,000 retweets.

- The above summary indicates that the vast majority of tweets have been retweeted a very low number of times. In the age of content amassing billions 
of views and millions of likes, intuitively, it can be said that only a modest handful of tweets in this dataset are truly 'viral'.

- However, working with a very small subset of the data may not be too fruitful, so a slightly looser, yet still reasonable definition of 'viral tweet' was chosen:

	'a viral tweet is one, which has been shared more than 10 times the user's followers count, for followers count equal to or greater than 100.'

- This would mean that for a user with 100 followers, a tweet, which receives at least 1000 retweets, would be labeled 'viral'.

- This definition can be tweaked arbitrarily to find one, which gives the most powerful results, however, there is no immediate clue as to how much this 
should be tweaked, if at all.

- There are three reasons why this definition was chosen:

1. For someone who has 3 followers, a tweet with 1000 retweets could be considered viral, while for someone who has 1 million followers, 1000 retweets could 
be considered as a less than popular tweet. Hence, the 'virality' of the tweet must in part depend on the user's follower's count.

2. There must be some base level, non-zero followers count, predominantly because the vast majority of users have a followers count of zero or close to zero, 
which would give a very low threshold for their tweet to be labeled 'viral'. The minimum followers count = 100 was chosen arbitrarily.

3. This definition yielded 860 viral tweets out of the 11099 total. Tightening the criteria further would further restrict the data and potentially limit the 
ability of the KNN model to learn effectively (underfitting).

- From here onwards, 'viral tweet' will be labeled as such according to this definition.

#### <u> Visualisations </u>

- Additional features were created to explore the data visually.

- Figure 2.1 shows a scatter plot of user followers count vs retweet count. Visually, a loose positive trend exists, however, analytically, the correlation 
coefficient was determined to be very small at 0.2914, indicating a very weak positive relationship between the two features.

- Figures 2.2, 2.3, and 2.4 show bar charts of average retweet count by number of hashtags, number of links, and number of user mentions in viral tweets 
respectively. It can be seen that the viral tweets with the highest retweet count tend to have zero hashtags, one link, and one user mention.

#### <u> KNN Classifier </u>

- The viral tweets labels column and ['user_followers_count', 'user_friends_count', 'full_character_count', 'num_of_words', 'num_of_hashtags', 'num_of_links', 
'num_of_user_mentions\'92] were isolated into separate labels and features datasets.

- Figure 3.1 shows how the value of KNN accuracy score varies with \'92k\'92 nearest neighbours. The peak score value happens at k = 16 and is
0.9157657657657657 when all of the features are included in the analysis, indicating that these features account for 91.58% of this KNN model's predictive power.

- Upon further investigation, it was discovered that the peak score value can be increased further by excluding user_friends_count feature from the model. 
In this case, the peak score value happens at k = 6 and is 0.9202702702702703, gaining 0.45% on the previous value, as can be seen in Figure 3.2.

- This was the highest possible score value for all possible combinations of the selected features.

- 92.03% accuracy is good, but it is not as high as the desired 95% accuracy. Further analysis is needed, possibly of the language used in tweets, to be able to 
determine 'viral' tweets with greater accuracy.

#### <u> Natural Language </u>

- There are multiple languages found in this tweets dataset:

['english', 'tagalog', 'japanese', 'finnish', 'korean', 'undefined', 'dutch', 'spanish', 'portuguese', 'indonesian', 'urdu', 'vietnamese', 'italian', 'thai', 
'czech', 'french', 'russian', 'greek', 'arabic', 'polish', 'romanian', 'turkish', 'slovenian', 'german', 'chinese', 'persian', 'swedish', 'estonian', 'hindi', 
'haitian creole']

- However, those which are in English comprise 95.71% of the total dataset and 96.63% of the viral tweets subset. It was decided that for the initial analysis, 
the focus will be only on the English tweets.

^^^ UNFINISHED SECTION

### CONCLUSIONS

- The maximum retweet count in the dataset is 413,719 with the average being 2,77.96 retweets. The median retweet count is 13. The number of tweets with a 
retweet count over 100,000 is 24.

- For a definition of 'viral' - 'a tweet, which has been shared more than 10 times the user's followers count, for followers count equal to or greater than 
100', there were 860 tweets, which were labeled 'viral'.

- There was a weak-positive to no statistically significant correlation between a user\'92s followers count and the retweet count they receive on their tweet, 
with Pearson\'92s correlation coefficient = 0.2914.

- Viral tweets with the highest average retweet count tend to have one or more of the following characteristics: zero hashtags, one link, and/ or one user mention.

- Using features: ['user_followers_count', 'full_character_count', 'num_of_words', 'num_of_hashtags', 'num_of_links', 'num_of_user_mentions'], a KNN model 
accuracy score of 92.03% for k = 6 was achieved. This was the highest possible score for all possible combinations of selected features.
