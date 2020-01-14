import sys
import pathlib as path
import numpy as np
import matplotlib.pyplot as plt



# # Loading the data
# We shuffle the data frame in case the classes would be sorted.
# This can be done with the **reindex** method applied on the **permutation** of the original indices.
# In this notebook we will only focus on the text variable and the class variable.
import pandas as pd
def LoadDataCsv(file_path:path, text_col:str, calss_col:str)->pd:
	pd.set_option('display.max_colwidth', -1)
	data_frame = pd.read_csv(file_path)
	data_frame = data_frame.reindex(np.random.permutation(data_frame.index))
	data_frame = data_frame[[text_col, calss_col]]
	return(data_frame)



# # Exploratory Data Analysis
### Target variable
# There are three class labels to predict: *negative, neutral or positive*.
# **CONCLUSION: **The class labels are **imbalanced** as we can see below.
# This is something that we should keep in mind during the model training phase.
# We could, for instance, make sure the classes are balanced by up/undersampling.
# In[31]:
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
def ShowDistribution(data_frame:pd, class_col:str, file_path:path):
	target_dist = sns.catplot(x=class_col, data=data_frame, kind="count", height=6, aspect=1.5, palette="PuBuGn_d")
	plt.show();
	if not file_path:
	    target_dist.savefig(file_path)



# ## Text variable
# To analyze the text variable we create a class **TextCounts**.
# In this class we compute some basic statistics on the text variable.
# This class can be used later in a Pipeline, as well.
# * **count_words** : number of words in the tweet
# * **count_mentions** : referrals to other Twitter accounts, which are preceded by a @
# * **count_hashtags** : number of tag words, preceded by a #
# * **count_capital_words** : number of uppercase words, could be used to *"shout"* and express (negative) emotions
# * **count_excl_quest_marks** : number of question or exclamation marks
# * **count_urls** : number of links in the tweet, preceded by http(s)
# * **count_emojis** : number of emoji, which might be a good indication of the sentiment
# In[3]:
import re
import emoji
from sklearn.base import BaseEstimator
from sklearn.base import  TransformerMixin
class TextCounts(BaseEstimator, TransformerMixin):

	def count_regex(self, pattern, tweet):
		return len(re.findall(pattern, tweet))

	def fit(self, X, y=None, **fit_params):
		# fit method is used when specific operations need to be done on the train data, but not on the test data
		return self

	def transform(self, X, **transform_params):
		count_words = X.apply(lambda x: self.count_regex(r'\w+', x))
		count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
		count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
		count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))
		count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', x))
		count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))

		# We will replace the emoji symbols with a description, which makes using a regex for counting easier
		# Moreover, it will result in having more words in the tweet
		count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))
		data_frame = pd.DataFrame({'count_words': count_words, 'count_mentions': count_mentions, 'count_hashtags': count_hashtags
			                  , 'count_capital_words': count_capital_words, 'count_excl_quest_marks': count_excl_quest_marks
			                  , 'count_urls': count_urls, 'count_emojis': count_emojis})
		return data_frame

def TextCountsRun(data_frame, text_col:str, class_col:str, out_path:path, show:bool):
	# In[4]:
	text_count = TextCounts()
	#data_frame_eda = text_count.fit_transform(data_frame['text'])
	data_frame_eda = text_count.fit_transform(data_frame[text_col])

	# Add airline_sentiment to data_fame_eda
	#data_frame_eda['airline_sentiment'] = data_frame['airline_sentiment']
	data_frame_eda[class_col] = data_frame[class_col]

	if show:
		show_dist_all(data_frame_eda, class_col, out_path)
	return(text_count, data_frame_eda)


# It could be interesting to see how the TextStats variables relate to the class variable.
# Therefore we write a function **show_dist** that provides descriptive statistics and a plot per target class.
# In[15]:
def show_dist(df, data_col, class_col:str, out_path:path, name:str):
    print('Descriptive stats for {}'.format(data_col))
    print('-' * (len(data_col) + 22))
    #print(data_frame.groupby('airline_sentiment')[data_col].describe())
    print(df.groupby(class_col)[data_col].describe())
    bins = np.arange(df[data_col].min(), df[data_col].max() + 1)
    #grid = sns.FacetGrid(data_frame, col='airline_sentiment', height=5, hue='airline_sentiment', palette="PuBuGn_d")
    grid = sns.FacetGrid(df, col=class_col, height=5, hue=class_col, palette="PuBuGn_d")
    grid = grid.map(sns.distplot, data_col, kde=False, norm_hist=True, bins=bins)
    plt.show()
    grid.savefig(out_path.joinpath(data_col + name))
    return;


# **CONCLUSIONS: **
# * **The number of words** used in the tweets is rather low.
# Maximum number of words is 36 and there are even tweets with only 2 words.
# So we'll have to be careful during data cleaning not to remove too many words.
# On the other hand, the text processing will be faster.
# Negative tweets contain more words than neutral or positive tweets.
# * All tweets have at least one **mention**.
# Probably this is the result of extracting the tweets based on mentions in the Twitter data.
# There seems to be no difference in number of mentions with regard to the sentiment.
# * Most of the tweets do not contain **hash tags**.
# So probably this variable will not be retained during model training.
# Again, no difference in number of hash tags with regard to the sentiment.
# * Most of the tweets do not contain **capitalized words** and we do not see a difference in distribution between the sentiments.
# * The positive tweets seem to be using a bit more **exclamation or question marks**.
# * Most tweets do not contain a **URL**.
# * Most tweets do not use **emojis**.
def show_dist_all(df_eda:list, class_col:str, out_path:path):
	# In[16]:
	show_dist(df_eda, 'count_words', class_col, out_path, '_dist.png')

	# In[17]:
	show_dist(df_eda, 'count_mentions', class_col, out_path, '_dist.png')

	# In[18]:
	show_dist(df_eda, 'count_hashtags', class_col, out_path, '_dist.png')

	# In[19]:
	show_dist(df_eda, 'count_capital_words', class_col, out_path, '_dist.png')

	# In[20]:
	show_dist(df_eda, 'count_excl_quest_marks', class_col, out_path, '_dist.png')

	# In[21]:
	show_dist(df_eda, 'count_urls', class_col, out_path, '_dist.png')

	# In[22]:
	show_dist(df_eda, 'count_emojis', class_col, out_path, '_dist.png')
	return;


# # Text Cleaning
# Before we start using the tweets' text we clean it.
# We'll do the this in the class CleanText:
# - remove the **mentions**, as we want to make the model generalisable to tweets of other airline companies too.
# - remove the **hash tag sign** (#) but not the actual tag as this may contain information
# - set all words to **lowercase**
# - remove all **punctuations**, including the question and exclamation marks
# - remove the **urls** as they do not contain useful information and we did not notice a distinction in the number of urls used between the sentiment classes.
# - make sure the converted **emojis** are kept as one word.
# - remove **digits**
# - remove **stopwords**
# - apply the **PorterStemmer** to keep the stem of the words.

# In[5]:
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
class CleanText(BaseEstimator, TransformerMixin):

	def remove_mentions(self, input_text):
		return re.sub(r'@\w+', '', input_text)

	def remove_urls(self, input_text):
		return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

	def emoji_oneword(self, input_text):
		# By compressing the underscore, the emoji is kept as one word
		return input_text.replace('_', '')

	def remove_punctuation(self, input_text):
		# Make translation table
		punct = string.punctuation
		trantab = str.maketrans(punct, len(punct) * ' ')  # Every punctuation symbol will be replaced by a space
		return input_text.translate(trantab)

	def remove_digits(self, input_text):
		return re.sub('\d+', '', input_text)

	def to_lower(self, input_text):
		return input_text.lower()

	def remove_stopwords(self, input_text):
		stopwords_list = stopwords.words('english')
		# Some words which might indicate a certain sentiment are kept via a whitelist
		whitelist = ["n't", "not", "no"]
		words = input_text.split()
		clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
		return " ".join(clean_words)

	def stemming(self, input_text):
		porter = PorterStemmer()
		words = input_text.split()
		stemmed_words = [porter.stem(word) for word in words]
		return " ".join(stemmed_words)

	def fit(self, X, y=None, **fit_params):
		return self

	def transform(self, X, **transform_params):
		clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(
			self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(
			self.stemming)
		return clean_X


		# To show how the cleaned text variable will look like, here's a sample.
		# In[6]:
def CleanTextRun(data_frame, text_col, out_path:path, show:bool):
	text_clean = CleanText().fit_transform(data_frame[text_col])
	text_clean.sample(5)

	text_clean = EmptyText(text_clean)
	if show:
		ShowFreqWords(text_clean, out_path.joinpath('bar_freq_word.png'))

	return (text_clean)


# **NOTE: **One side-effect of text cleaning is that some rows do not have any words left in their text.
# For the CountVectorizer and TfIdfVectorizer this does not really pose a problem.
# However, for the Word2Vec algorithm this causes an error.
# There are different strategies that you could apply to deal with these missing values.
#
# * Remove the complete row, but in a production environment this is not really desirable.
# * Impute the missing value with some placeholder text like *[no_text]*
# * Word2Vec: use the average of all vectors
#
# Here we will impute with a placeholder text.

# In[7]:
def EmptyText(text_clean):
	empty_clean = text_clean == ''
	print('{} records have no words left after text cleaning'.format(text_clean[empty_clean].count()))
	text_clean.loc[empty_clean] = '[no_text]'
	return(text_clean)


# Now that we have the cleaned text of the tweets, we can have a look at what are the most frequent words.
# Below we'll show the top 20 words.
#
# **CONCLUSION: **Not surprisingly the most frequent word is *flight*.
# In[29]:
import collections
from sklearn.feature_extraction.text import CountVectorizer
def ShowFreqWords(text_clean, out_file:path):
	cv = CountVectorizer()
	bow = cv.fit_transform(text_clean)
	word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
	word_counter = collections.Counter(word_freq)
	word_counter_df = pd.DataFrame(word_counter.most_common(20), columns=['word', 'freq'])

	fig, ax = plt.subplots(figsize=(12, 10))
	bar_freq_word = sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
	plt.show();
	bar_freq_word.get_figure().savefig(out_file)
	return;


# # Creating test data
# To evaluate the trained models we'll need a **test set**.
# Evaluating on the train data would not be correct because the models are trained to minimize their cost function.
#
# First we combine the TextCounts variables with the CleanText variable.
# **NOTE: **Initially, I made the mistake to do execute TextCounts and CleanText in the GridSearchCV below.
# This took too long as it applies these functions each run of the GridSearch.
# It suffices to run them only once.

# In[8]:
from sklearn.model_selection import train_test_split
def CreateTest(df_eda, text_clean):
	data_frame_model = df_eda
	data_frame_model['clean_text'] = text_clean
	data_frame_model.columns.tolist()
	return(data_frame_model)



# So df_model now contains several variables. However, our vectorizers (see below) will only need the *clean_text* variable.
# The TextCounts variables can be added as such. To specifically select columns, I wrote the class **ColumnExtractor** below.
# This can be used in the Pipeline afterwards.
# In[9]:
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, **transform_params):
        return X[self.cols]

    def fit(self, X, y=None, **fit_params):
        return self



### Hyperparameter tuning and cross-validation
# As we will see below, the vectorizers and classifiers all have configurable parameters.
# In order to chose the best parameters, we need to evaluate on a separate validation set that was not used during the training.
# However, using only one validation set may not produce reliable validation results. Due to chance you might have a good model performance on the validation set.
# If you would split the data otherwise, you might end up with other results. To get a more accurate estimation, we perform **cross-validation**.
#
# With cross-validation the data is split into a train and validation set multiple times.
# The evaluation metric is then averaged over the different folds. Luckily, GridSearchCV applies cross-validation out-of-the-box.
#
# To find the best parameters for both a vectorizer and classifier, we create a **Pipeline**.
# All this is put into a function for ease of use.
###

# ### Evaluation metrics
# By default GridSearchCV uses the default scorer to compute the *best_score_*.
# For both the MultiNomialNb and LogisticRegression this default scoring metric is the accuracy.
#
# In our function *grid_vect* we additionally generate the *classification_report* on the test data.
# This provides some interesting metrics **per target class**, which might be more appropriate here.
# These metrics are the **precision, recal and F1 score.**
#
# * **Precision: ** Of all rows we predicted to be a certain class, how many did we correctly predict?
# * **Recall: ** Of all rows of a certain class, how many did we correctly predict?
# * **F1 score: ** Harmonic mean of Precision and Recall.
#
# Precision and Recall can be calculated with the elements of the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

# In[18]:
SIZE = 25
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.externals import joblib
# Based on http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
def grid_vect(clf, parameters_clf, X_train, X_test, y_train, y_test, count_cols, out_file:path, parameters_text=None, vect=None, is_w2v=False) -> ColumnExtractor:

	colExtractor = ColumnExtractor(cols=count_cols)
	if is_w2v:
		w2vcols = []
		for i in range(SIZE):
			w2vcols.append(i)
		features = FeatureUnion([('textcounts', colExtractor), ('w2v', ColumnExtractor(cols=w2vcols))], n_jobs=-1)
	else:
		features = FeatureUnion([('textcounts', colExtractor),('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text')), ('vect', vect)]))], n_jobs=-1)

	pipeline = Pipeline([('features', features), ('clf', clf)])

	# Join the parameters dictionaries together
	parameters = dict()
	if parameters_text:
		parameters.update(parameters_text)
	parameters.update(parameters_clf)

	# Make sure you have scikit-learn version 0.19 or higher to use multiple scoring metrics
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)

	print("Performing grid search...")
	print("pipeline:", [name for name, _ in pipeline.steps])
	print("parameters:")
	print(parameters)

	t0 = time()
	grid_search.fit(X_train, y_train)
	print("done in %0.3fs" % (time() - t0))
	print()

	print("Best CV score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
	print("\n")
	print("Classification Report Test Data")
	print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))

	if out_file:
		joblib.dump(colExtractor, out_file)

	return grid_search




# ### Parameter grids for GridSearchCV
# In[11]:
# Parameter grid settings for the vectorizers (Count and TFIDF)
parameters_vect = {
    'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
    'features__pipe__vect__ngram_range': ((1, 1), (1, 2)),
    'features__pipe__vect__min_df': (1,2)
}


# Parameter grid settings for MultinomialNB
parameters_mnb = {
    'clf__alpha': (0.25, 0.5, 0.75)
}


# Parameter grid settings for LogisticRegression
parameters_logreg = {
    'clf__C': (0.25, 0.5, 1.0),
    'clf__penalty': ('l1', 'l2')
}


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
def BestContVect(X_train, X_test, y_train, y_test, count_cols, out_path:path):
	# ## Classifiers
	# Here we will compare the performance of a MultinomailNB and LogisticRegression.
	# In[20]:
	mnb = MultinomialNB()
	logreg = LogisticRegression()

	# ## CountVectorizer
	# To use words in a classifier, we need to convert the words to numbers.
	# This can be done with a CountVectorizer.
	# Sklearn's **CountVectorizer** takes all words in all tweets, assigns an ID and counts the frequency of the word per tweet.
	# This *bag of words* can then be used as input for a classifier.
	# It is what is called a **sparse** data set, meaning that each record will have many zeroes for the words not occurring in the tweet.
	# In[38]:
	countvect = CountVectorizer()

	# In[40]:
	# MultinomialNB
	best_mnb_countvect = grid_vect(mnb, parameters_mnb, X_train, X_test, y_train, y_test, count_cols, out_path.joinpath('best_mnb_countvect.pkl'), parameters_text=parameters_vect, vect=countvect)

	# In[41]:
	# LogisticRegression
	best_logreg_countvect = grid_vect(logreg, parameters_logreg, X_train, X_test, y_train, y_test, count_cols, out_path.joinpath('best_logreg_countvect.pkl'), parameters_text=parameters_vect, vect=countvect)

	return(best_mnb_countvect, best_logreg_countvect);


def BestTfIdf(X_train, X_test, y_train, y_test, count_cols, out_path:path):
	# ## Classifiers
	# Here we will compare the performance of a MultinomailNB and LogisticRegression.
	# In[20]:
	mnb = MultinomialNB()
	logreg = LogisticRegression()


	# ## TF-IDF
	# One issue with CountVectorizer is that there might be words that occur frequently in observations of the target classes.
	# These words do not have discriminatory information and can be removed.
	# [TF-IDF (term frequency - inverse document frequency)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) can be used to downweight these frequent words.
	# In[42]:
	tfidfvect = TfidfVectorizer()

	# In[43]:
	# MultinomialNB
	best_mnb_tfidf = grid_vect(mnb, parameters_mnb, X_train, X_test, y_train, y_test, count_cols, out_path.joinpath('best_mnb_tfidf.pkl'), parameters_text=parameters_vect, vect=tfidfvect)

	# In[45]:
	# LogisticRegression
	best_logreg_tfidf = grid_vect(logreg, parameters_logreg, X_train, X_test, y_train, y_test, count_cols, out_path.joinpath('best_logreg_tfidf.pkl'), parameters_text=parameters_vect, vect=tfidfvect)

	return(best_mnb_tfidf, best_logreg_tfidf);



# ## Word2Vec
# Another way of converting the words in the tweets to numerical values can be achieved with Word2Vec.
# Word2Vec maps each word in a multi-dimensional space.
# It does this by taking into account the context in which a word appears in the tweets.
# As a result, words that are semantically similar are also close to each other in the multi-dimensional space.
#
# The Word2Vec algorithm is implemented in the [gensim](https://radimrehurek.com/gensim/models/word2vec.html) package.
#
import gensim
from nltk.tokenize import word_tokenize
def BestWord2Vec(X_train, X_test, y_train, y_test, count_cols, out_path:path):
	# The Word2Vec algorithm uses lists of words as input.
	# For that purpose we use the **word_tokenize** method of the the nltk package.
	# In[27]:
	SIZE = 25
	X_train['clean_text_wordlist'] = X_train.clean_text.apply(lambda x: word_tokenize(x))
	X_test['clean_text_wordlist'] = X_test.clean_text.apply(lambda x: word_tokenize(x))
	model = gensim.models.Word2Vec(X_train.clean_text_wordlist, min_count=1, size=SIZE, window=3, workers=4)

	# In[28]:
	model.most_similar('plane', topn=3)


	# The Word2Vec model provides a vocabulary of the words in the corpus together with their vector values.
	# The number of vector values is equal to the chosen **size**.
	# These are the dimensions on which each word is mapped in the multi-dimensional space.
	#
	# Words with an occurrence less than **min_count** are not kept in the vocabulary.
	# **NOTE: **A side effect of the **min_count** parameter is that some tweets could have no vector values.
	# This is would be the case when the word(s) in the tweet occur in less than *min_count* tweets.
	# Due to the small corpus of tweets, there is a risk of this happening in our case. Therefore we set the min_count value equal to 1.
	#
	# The tweets can have a different number of vectors, depending on the number of words it contains.
	# To use this output for modeling we will aggregate the vectors per tweet to have the same number (i.e. *size*) of input variables per tweet.
	# Therefore we will take the average of all vectors per tweet. We do this with the function **compute_avg_w2v_vector**.
	# In this function we also check whether the words in the tweet occur in the vocabulary of the word2vec model. If not, a list filled with 0.0 is returned.
	# Else the average of the word vectors.
	# In[14]:

	def compute_avg_w2v_vector(w2v_dict, tweet):
		list_of_word_vectors = [w2v_dict[w] for w in tweet if w in w2v_dict.vocab.keys()]

		if len(list_of_word_vectors) == 0:
			result = [0.0] * SIZE
		else:
			result = np.sum(list_of_word_vectors, axis=0) / len(list_of_word_vectors)

		return result

	# In[29]:
	X_train_w2v = X_train['clean_text_wordlist'].apply(lambda x: compute_avg_w2v_vector(model.wv, x))
	X_test_w2v = X_test['clean_text_wordlist'].apply(lambda x: compute_avg_w2v_vector(model.wv, x))

	# This gives us a Series with a vector of dimension equal to SIZE.
	# Now we will split this vector and create a DataFrame with each vector value in a separate column.
	# That way we can concatenate the word2vec variables to the other TextCounts variables.
	# We need to reuse the index of X_train and X_test respectively. Otherwise this will give issues (duplicates) in the concatenation later on.
	# In[30]:
	X_train_w2v = pd.DataFrame(X_train_w2v.values.tolist(), index= X_train.index)
	X_test_w2v = pd.DataFrame(X_test_w2v.values.tolist(), index= X_test.index)

	# Concatenate with the TextCounts variables
	X_train_w2v = pd.concat([X_train_w2v, X_train.drop(['clean_text', 'clean_text_wordlist'], axis=1)], axis=1)
	X_test_w2v = pd.concat([X_test_w2v, X_test.drop(['clean_text', 'clean_text_wordlist'], axis=1)], axis=1)


	# **NOTE: **We only consider LogisticRegression as we have negative values in the Word2Vec vectors.
	# MultinomialNB assumes that the variables have a [multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution) which cannot contain negative values.

	# In[31]:
	logreg = LogisticRegression()
	best_logreg_w2v = grid_vect(logreg, parameters_logreg, X_train_w2v, X_test_w2v,  y_train, y_test, count_cols, out_path.joinpath('best_logreg_w2v.pkl'), is_w2v=True)

	return(best_logreg_w2v)


# ## Conclusion
# * Both classifiers achieve the best results when using the features of the CountVectorizer
# * Overall, Logistic Regression outperforms the Multinomial Naive Bayes classifier
# * The best performance on the test set comes from the LogisticRegression with features from CountVectorizer.
#
# Best parameters:
# * C value of 1
# * L2 regularization
# * max_df: 0.5 or maximum document frequency of 50%.
# * min_df: 1 or the words need to appear in at least 2 tweets
# * ngram_range: (1, 2), both single words as bi-grams are used
#
# Evaluation metrics:
# * A **test accuracy** of 81,3%, which is better than what we would achieve by setting the prediction for all observations to the majority class (*negative* which would give 63% accuracy).
# * The **Precision** is rather high for all three classes. For instance, of all cases that we predict as negative, 80% is indeed negative.
# * The **Recall** for the neutral class is low. Of all neutral cases in our test data, we only predict 48% as being neutral.

# # Apply the best model on new tweets
# Just for the fun we will use the best model and apply it to some new tweets that contain *@VirginAmerica*.
# I selected 3 negative and 3 positive tweets by hand.
#
# Thanks to the GridSearchCV, we now know what are the best hyperparameters.
# So now we can train the best model on **all training data**, including the test data that we split off before.
def Predict(new_tweets:list, df_model, class_col:str, text_count, count_cols:list):

	features = FeatureUnion([     ('textcounts', ColumnExtractor(cols=count_cols))
		                        ,  ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text'))
			                     ,  ('vect', CountVectorizer(max_df=0.5, min_df=1, ngram_range=(1, 2)))]))], n_jobs=-1)

	pipeline = Pipeline([('features', features), ('clf', LogisticRegression(C=1.0, penalty='l2'))])

	best_model = pipeline.fit(df_model.drop(class_col, axis=1), df_model[class_col])

	df_counts_pos = text_count.transform(new_tweets)
	df_clean_pos = CleanText().transform(new_tweets)

	df_model_pos = df_counts_pos
	df_model_pos['clean_text'] = df_clean_pos
	pos_predict = best_model.predict(df_model_pos).tolist()

	print(pos_predict)
	return










CLASS_COLUMN = 'airline_sentiment'
TEXT_COLUMN = 'text'
class SentimetTwits:
	def __init__(self, show:bool, dump:bool):
		self._show =  show
		self._dump = dump
		return

	def Run(self, file_path:path, out_path:path):
		data_frame = LoadDataCsv(file_path, TEXT_COLUMN, CLASS_COLUMN)
		if self._show:
			ShowDistribution(data_frame, CLASS_COLUMN, out_path.joinpath('target_dist.png'))

		# In[4]:
		text_count, data_fame_eda = TextCountsRun(data_frame, TEXT_COLUMN, CLASS_COLUMN, out_path, self._show)

		# To show how the cleaned text variable will look like, here's a sample.
		# In[6]:
		text_clean = CleanTextRun(data_frame, TEXT_COLUMN, out_path, self._show)

		data_frame_model = CreateTest(data_fame_eda, text_clean)

		# In[10]:
		X_train, X_test, y_train, y_test = train_test_split(data_frame_model.drop(CLASS_COLUMN, axis=1), data_frame_model[CLASS_COLUMN], test_size=0.1, random_state=37)

		file_path = None
		if(self._dump):   file_path = out_path
		count_cols = ['count_capital_words', 'count_emojis', 'count_excl_quest_marks', 'count_hashtags',
		                  'count_mentions', 'count_urls', 'count_words']
		best_mnb_countvect, best_logreg_countvect= BestContVect(X_train, X_test, y_train, y_test, count_cols, file_path)
		best_mnb_tfidf, best_logreg_tfidf = BestTfIdf(X_train, X_test, y_train, y_test, count_cols, file_path)
		best_logreg_w2v = BestWord2Vec(X_train, X_test, y_train, y_test, count_cols, file_path)


		# In[32]:
		# ## New positive tweets
		# In[33]:
		new_positive_tweets = pd.Series([
			 "Thank you @VirginAmerica for you amazing customer support team on Tuesday 11/28 at @EWRairport and returning my lost bag in less than 24h! #efficiencyiskey #virginamerica"
			,"Love flying with you guys ask these years.  Sad that this will be the last trip 😂   @VirginAmerica  #LuxuryTravel"
			,"Wow @VirginAmerica main cabin select is the way to fly!! This plane is nice and clean & I have tons of legroom! Wahoo! NYC bound! ✈️"])
		Predict(new_positive_tweets, data_frame_model, CLASS_COLUMN, text_count, count_cols)

		# ## New negative tweets
		# In[34]:
		new_negative_tweets = pd.Series([
			 "@VirginAmerica shocked my initially with the service, but then went on to shock me further with no response to what my complaint was. #unacceptable @Delta @richardbranson"
			,"@VirginAmerica this morning I was forced to repack a suitcase w a medical device because it was barely overweight - wasn't even given an option to pay extra. My spouses suitcase then burst at the seam with the added device and had to be taped shut. Awful experience so far!"
			,"Board airplane home. Computer issue. Get off plane, traverse airport to gate on opp side. Get on new plane hour later. Plane too heavy. 8 volunteers get off plane. Ohhh the adventure of travel ✈️ @VirginAmerica"])

		Predict(new_negative_tweets, data_frame_model, CLASS_COLUMN, text_count, count_cols)

		return



from  PyBase import Files

def main(argv):
	sentiments = SentimetTwits(show=False, dump=True)
	if(Files.DirNew('./output')):
		sentiments.Run(path.Path('./input/Tweets.csv'), path.Path('./OutSmall'))
	return;

if __name__ == '__main__':
	main(sys.argv)