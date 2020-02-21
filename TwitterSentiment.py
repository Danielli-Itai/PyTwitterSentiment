import sys
import pandas
import pathlib as path
import numpy as np

sys.path.append('../PyBaseNlp')
from PyBaseNlp import DataPlot
from PyBaseNlp import DataTools
from PyBaseNlp import TextNorm
from PyBaseNlp import TextTools





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
SIZE = 25
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.externals import joblib
# Based on http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
def grid_vect(clf, parameters_clf, X_train, X_test, y_train, y_test, count_cols, out_file:path, parameters_text=None, vect=None, is_w2v=False)->TextTools.ColumnExtractor:

	colExtractor = TextTools.ColumnExtractor(cols=count_cols)
	if is_w2v:
		w2vcols = []
		for i in range(SIZE):
			w2vcols.append(i)
		features = FeatureUnion([('textcounts', colExtractor), ('w2v', TextTools.ColumnExtractor(cols=w2vcols))], n_jobs=-1)
	else:
		features = FeatureUnion([('textcounts', colExtractor),('pipe', Pipeline([('cleantext', TextTools.ColumnExtractor(cols='clean_text')), ('vect', vect)]))], n_jobs=-1)

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





from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
def BestContVect(X_train, X_test, y_train, y_test, count_cols, parameters_mnb:dict, parameters_logreg:dict, parameters_vect:dict, out_path:path):
	# ## Classifiers
	# Here we will compare the performance of a MultinomailNB and LogisticRegression.
	mnb = MultinomialNB()
	logreg = LogisticRegression()

	# ## CountVectorizer
	# To use words in a classifier, we need to convert the words to numbers.
	# This can be done with a CountVectorizer.
	# Sklearn's **CountVectorizer** takes all words in all tweets, assigns an ID and counts the frequency of the word per tweet.
	# This *bag of words* can then be used as input for a classifier.
	# It is what is called a **sparse** data set, meaning that each record will have many zeroes for the words not occurring in the tweet.
	countvect = CountVectorizer()

	# MultinomialNB
	print('***** BestContVect MultinomialNB')
	best_mnb_countvect = grid_vect(mnb, parameters_mnb, X_train, X_test, y_train, y_test, count_cols, out_path.joinpath('best_mnb_countvect.pkl'), parameters_text=parameters_vect, vect=countvect)

	# LogisticRegression
	print('***** BestContVect LogisticRegression')
	best_logreg_countvect = grid_vect(logreg, parameters_logreg, X_train, X_test, y_train, y_test, count_cols, out_path.joinpath('best_logreg_countvect.pkl'), parameters_text=parameters_vect, vect=countvect)

	return(best_mnb_countvect, best_logreg_countvect);


def BestTfIdf(X_train, X_test, y_train, y_test, count_cols,  parameters_mnb:dict, parameters_logreg:dict, parameters_vect:dict, out_path:path):
	# ## Classifiers
	# Here we will compare the performance of a MultinomailNB and LogisticRegression.
	mnb = MultinomialNB()
	logreg = LogisticRegression()

	# ## TF-IDF
	# One issue with CountVectorizer is that there might be words that occur frequently in observations of the target classes.
	# These words do not have discriminatory information and can be removed.
	# [TF-IDF (term frequency - inverse document frequency)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) can be used to downweight these frequent words.
	tfidfvect = TfidfVectorizer()

	# MultinomialNB
	print('***** BestTfIdf MultinomialNB')
	best_mnb_tfidf = grid_vect(mnb, parameters_mnb, X_train, X_test, y_train, y_test, count_cols, out_path.joinpath('best_mnb_tfidf.pkl'), parameters_text=parameters_vect, vect=tfidfvect)

	# LogisticRegression
	print('***** BestTfIdf LogisticRegression')
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
def BestWord2Vec(X_train, X_test, y_train, y_test, count_cols, parameters_logreg:dict, out_path:path):
	# The Word2Vec algorithm uses lists of words as input.
	# For that purpose we use the **word_tokenize** method of the the nltk package.
	SIZE = 25
	X_train['clean_text_wordlist'] = X_train.clean_text.apply(lambda x: word_tokenize(x))
	X_test['clean_text_wordlist'] = X_test.clean_text.apply(lambda x: word_tokenize(x))
	model = gensim.models.Word2Vec(X_train.clean_text_wordlist, min_count=1, size=SIZE, window=3, workers=4)
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
	# To use this output for modeling we will aggregate the vectors per tweet to have the same number (i.e. *size*) of input statistics per tweet.
	# Therefore we will take the average of all vectors per tweet. We do this with the function **compute_avg_w2v_vector**.
	# In this function we also check whether the words in the tweet occur in the vocabulary of the word2vec model. If not, a list filled with 0.0 is returned.
	# Else the average of the word vectors.
	def compute_avg_w2v_vector(w2v_dict, tweet):
		list_of_word_vectors = [w2v_dict[w] for w in tweet if w in w2v_dict.vocab.keys()]

		if len(list_of_word_vectors) == 0:
			result = [0.0] * SIZE
		else:
			result = np.sum(list_of_word_vectors, axis=0) / len(list_of_word_vectors)

		return result

	X_train_w2v = X_train['clean_text_wordlist'].apply(lambda x: compute_avg_w2v_vector(model.wv, x))
	X_test_w2v = X_test['clean_text_wordlist'].apply(lambda x: compute_avg_w2v_vector(model.wv, x))

	# This gives us a Series with a vector of dimension equal to SIZE.
	# Now we will split this vector and create a DataFrame with each vector value in a separate column.
	# That way we can concatenate the word2vec statistics to the other TweetsCounts statistics.
	# We need to reuse the index of X_train and X_test respectively. Otherwise this will give issues (duplicates) in the concatenation later on.
	X_train_w2v = pandas.DataFrame(X_train_w2v.values.tolist(), index= X_train.index)
	X_test_w2v = pandas.DataFrame(X_test_w2v.values.tolist(), index= X_test.index)

	# Concatenate with the TweetsCounts statistics
	X_train_w2v = pandas.concat([X_train_w2v, X_train.drop(['clean_text', 'clean_text_wordlist'], axis=1)], axis=1)
	X_test_w2v = pandas.concat([X_test_w2v, X_test.drop(['clean_text', 'clean_text_wordlist'], axis=1)], axis=1)


	# **NOTE: **We only consider LogisticRegression as we have negative values in the Word2Vec vectors.
	# MultinomialNB assumes that the statistics have a [multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution) which cannot contain negative values.
	logreg = LogisticRegression()
	print('***** BestWord2Vec LogisticRegression')
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
from sklearn.feature_extraction.text import CountVectorizer
def Predict(new_tweets:list, data_frame_model, class_col:str, count_cols:list):
	start_time = datetime.now()
	features = FeatureUnion([     ('textcounts', TextTools.ColumnExtractor(cols=count_cols))
		                        ,  ('pipe', Pipeline([('cleantext', TextTools.ColumnExtractor(cols='clean_text'))
			                     ,  ('vect', CountVectorizer(max_df=0.5, min_df=1, ngram_range=(1, 2)))]))], n_jobs=-1)

	pipeline = Pipeline([('features', features), ('clf', LogisticRegression(C=1.0, penalty='l2'))])

	best_model = pipeline.fit(data_frame_model.drop(class_col, axis=1), data_frame_model[class_col])

	df_counts_pos = TextTools.TweetsCounts().transform(new_tweets)
	df_clean_pos =  TextNorm.TweetsClean().transform(new_tweets)

	df_model_pos = df_counts_pos
	df_model_pos['clean_text'] = df_clean_pos
	pos_predict = best_model.predict(df_model_pos).tolist()
	print("***** Predict Time = ", (datetime.now() - start_time))
	return(pos_predict)









from sklearn.model_selection import train_test_split
class SentimetTwits:
	def __init__(self, dump:bool):
		self._dump = dump
		self.count_cols = ['count_capital_words', 'count_emojis', 'count_excl_quest_marks', 'count_hashtags', 'count_mentions', 'count_urls', 'count_words']
		return


	def TextPrepare(self, data_frame, text_col:str, lable_col:str, show:bool, out_path:path):
		data_frame = data_frame[[text_col, lable_col]]
		data_frame_eda = TextTools.TextCountsRun(data_frame, text_col, lable_col)

		# To show how the cleaned text variable will look like, here's a sample.
		text_clean = TextNorm.CleanTextRun(data_frame, text_col, '[no_text]')

		return(data_frame_eda, text_clean)


	def TextPlots(self, data_frame, data_frame_eda, text_clean, text_col, lable_col:str, out_path:path):
		DataPlot.ShowDistribution(data_frame, lable_col, out_path.joinpath('target_dist.png'))
		#show_dist_all(data_frame_eda, lable_col, out_path)
		DataPlot.show_dist(data_frame_eda, 'count_words', lable_col, out_path, '_dist.png')
		DataPlot.show_dist(data_frame_eda, 'count_mentions', lable_col, out_path, '_dist.png')
		DataPlot.show_dist(data_frame_eda, 'count_hashtags', lable_col, out_path, '_dist.png')
		DataPlot.show_dist(data_frame_eda, 'count_capital_words', lable_col, out_path, '_dist.png')
		DataPlot.show_dist(data_frame_eda, 'count_excl_quest_marks', lable_col, out_path, '_dist.png')
		DataPlot.show_dist(data_frame_eda, 'count_urls', lable_col, out_path, '_dist.png')
		DataPlot.show_dist(data_frame_eda, 'count_emojis', lable_col, out_path, '_dist.png')

		word_counter_df = TextTools.ShowFreqWords(text_clean, out_path.joinpath('bar_freq_word.png'))
		DataPlot.ShowFrequency(word_counter_df, out_path.joinpath('bar_freq_word.png'))
		return

	def Train(self, data_fame_eda, text_clean, lable_col:str, out_path):
		self.data_frame_model = TextNorm.TextCleanSet(data_fame_eda, text_clean, 'clean_text')

		# In[10]:
		X_train, X_test, y_train, y_test = train_test_split(self.data_frame_model.drop(lable_col, axis=1), self.data_frame_model[lable_col], test_size=0.1, random_state=37)

		file_path = None
		if(self._dump):   file_path = out_path

		# ### Parameter grids for GridSearchCV
		# Parameter grid settings for the vectorizers (Count and TFIDF)
		parameters_vect = {
			'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
			'features__pipe__vect__ngram_range': ((1, 1), (1, 2)),
			'features__pipe__vect__min_df': (1, 2)
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
		best_mnb_countvect, best_logreg_countvect= BestContVect(X_train, X_test, y_train, y_test, self.count_cols, parameters_mnb, parameters_logreg, parameters_vect, file_path)
		best_mnb_tfidf, best_logreg_tfidf = BestTfIdf(X_train, X_test, y_train, y_test, self.count_cols, parameters_mnb, parameters_logreg, parameters_vect, file_path)
		best_logreg_w2v = BestWord2Vec(X_train, X_test, y_train, y_test, self.count_cols, parameters_logreg, file_path)
		return


	def Predict(self, text_series:pandas.Series, lable_col):
		return(Predict(text_series, self.data_frame_model, lable_col, self.count_cols))
		return


def RunTrain(file_path:path, file_name:str, text_col:str, lable_col:str, out_path:path):
	print("\n\n\n****************Training ", file_name, '****************')
	start_time = datetime.now()

	if not (Files.DirNew(out_path.joinpath('Train'+file_name))):  return(None)

	sentiments = SentimetTwits(dump=True)
	data_frame = DataTools.DataCsvLoad(file_path.joinpath(file_name), scrumbele=True)

	data_frame_eda, text_clean = sentiments.TextPrepare(data_frame, text_col, lable_col, True, out_path.joinpath('Train'+file_name))
	sentiments.TextPlots(data_frame, data_frame_eda, text_clean, text_col, lable_col, out_path.joinpath('Train'+file_name))
	sentiments.Train(data_frame_eda, text_clean, lable_col, out_path.joinpath('Train'+file_name))

	## New positive tweets
	new_positive_tweets = pandas.Series([
		 "Thank you @VirginAmerica for you amazing customer support team on Tuesday 11/28 at @EWRairport and returning my lost bag in less than 24h! #efficiencyiskey #virginamerica"
		,"Love flying with you guys ask these years.  Sad that this will be the last trip 😂   @VirginAmerica  #LuxuryTravel"
		,"Wow @VirginAmerica main cabin select is the way to fly!! This plane is nice and clean & I have tons of legroom! Wahoo! NYC bound! ✈️"])
	pos_predict = sentiments.Predict(new_positive_tweets, lable_col)
	print(pos_predict)

	## New negative tweets
	new_negative_tweets = pandas.Series([
		 "@VirginAmerica shocked my initially with the service, but then went on to shock me further with no response to what my complaint was. #unacceptable @Delta @richardbranson"
		,"@VirginAmerica this morning I was forced to repack a suitcase worker_lock a medical device because it was barely overweight - wasn't even given an option to pay extra. My spouses suitcase then burst at the seam with the added device and had to be taped shut. Awful experience so far!"
		,"Board airplane home. Computer issue. Get off plane, traverse airport to gate on opp side. Get on new plane hour later. Plane too heavy. 8 volunteers get off plane. Ohhh the adventure of travel ✈️ @VirginAmerica"])
	pos_predict = sentiments.Predict(new_negative_tweets, lable_col)
	print(pos_predict)

	print("**************** Total Time = ", (datetime.now() - start_time), '****************\n\n\n')
	return(sentiments)






def RunPredict(sentiments:SentimetTwits, in_path:path, file_name:str, text_col:str, lable_col:str, out_path:path)->pandas:
	print("\n\n\n**************** Predicting ", file_name, '****************')
	start_time = datetime.now()

	if not (Files.DirNew(out_path)):  return (None)
	data_frame = DataTools.DataCsvLoad(in_path.joinpath(file_name), False)

	print('Predicting ' + file_name)
	pos_predict = sentiments.Predict(data_frame[text_col], lable_col)

	print('Reporting' + file_name)
	data_frame[lable_col] = pos_predict
	DataTools.DataCsvSave(data_frame, out_path.joinpath(file_name))

	print("**************** Total Time = ", (datetime.now() - start_time), '****************\n\n\n')
	return (data_frame)

def RunReport(data_frame:pandas, text_col:str, lable_col:str, out_path:path):
	print("\n\n\n**************** Reporting ", '****************')
	start_time = datetime.now()

	if not (Files.DirNew(out_path)):  return (None)
	sentiments = SentimetTwits(True)
	data_frame_eda1, text_clean1 = sentiments.TextPrepare(data_frame, text_col=text_col, lable_col=lable_col, show=True, out_path=out_path)
	sentiments.TextPlots(data_frame, data_frame_eda1, text_clean1, text_col, lable_col, out_path)
	print("**************** Total Time = ", (datetime.now() - start_time), '****************\n\n\n')
	return







import os
def RunFilePredict(sentiments:SentimetTwits, in_path:path, file_name:str, split_size:int, text_col:str, lable_col:str, out_path:path)->pandas.DataFrame:
	print("\n\n\n**************** File Predicting ", file_name, '****************')
	start_time = datetime.now()
	if not (Files.DirNew(out_path)):  return (None)

	tmp_split:path = path.Path(out_path.joinpath('./tmp_split'))
	tmp_merge: path = path.Path(out_path.joinpath('./tmp_merge'))
	if(not os.path.isdir(tmp_split)):
		DataTools.DataCsvSplit(in_path, file_name, split_size, tmp_split)
		Files.DirDelete(tmp_merge)
		if not (Files.DirNew(tmp_merge)):  return (None)

	for name in Files.DirFiles(tmp_split):
		name_path = path.Path(name)
		data_frame = DataTools.DataCsvLoad(name, False)
		text_clean = TextNorm.CleanTextRun(data_frame, text_col, '[no_text]')
		pos_predict = sentiments.Predict(text_clean, lable_col)
		data_frame[lable_col] = pos_predict
		data_frame[text_col] = text_clean
		DataTools.DataCsvSave(data_frame, tmp_merge.joinpath(name_path.name))
		Files.FileDelete(name)


	DataTools.DataCsvMerge(tmp_merge, file_name, out_path)
	Files.DirDelete(tmp_split)
	Files.DirDelete(tmp_merge)

	print("**************** Total Time = ", (datetime.now() - start_time), '****************\n\n\n')
	return


def RunFileReport(in_path:path, file_name:str, text_col:str, lable_col:str, out_path:path):
	print("\n\n\n**************** File Reporting ", file_name, '****************')
	start_time = datetime.now()

	if not (Files.DirNew(out_path)):  return (None)
	data_frame =  DataTools.DataCsvLoad(in_path.joinpath(file_name), False)
	sentiments = SentimetTwits(True)
	data_frame_eda1, text_clean1 = sentiments.TextPrepare(data_frame, text_col=text_col, lable_col=lable_col, show=True, out_path=out_path)
	sentiments.TextPlots(data_frame, data_frame_eda1, text_clean1, text_col, lable_col, out_path)
	print("**************** Total Time = ", (datetime.now() - start_time), '****************\n\n\n')
	return





def RunTweetsSentiment(text_col:str, class_col, train_file:str, predict_file:str):
	sentiments: SentimetTwits = RunTrain(path.Path('./input'), train_file, text_col, class_col, path.Path('./output'))
	data_frame:pandas = RunPredict(sentiments, path.Path('./input'), predict_file, text_col, class_col, path.Path('./output').joinpath(predict_file))
	RunReport(data_frame, text_col, class_col, path.Path('./output').joinpath(predict_file))
	return


def RunTweetsSentimentLarge(text_col:str, class_col, train_file:str, predict_file:str, split_size:int):
	sentiments: SentimetTwits = RunTrain(path.Path('./input'), train_file, text_col, class_col, path.Path('./output'))
	RunFilePredict(sentiments, path.Path('./input'), predict_file, split_size, text_col, class_col, path.Path('./output').joinpath(predict_file))
	RunFileReport(path.Path('./output').joinpath(predict_file), predict_file, text_col, class_col, path.Path('./output').joinpath(predict_file))





from  PyBase import Files
from datetime import datetime
def main(argv):
	TEXT_COL:str = 'text'
	SENTIMENT_COL:str = 'sentiment'

	TRAIN_SMALL:str = 'TweetsSmall.csv'
	TRAIN_FULL:str = 'TweetsFull.csv'
#	RunTweetsSentiment(TEXT_COL, SENTIMENT_COL, TRAIN_SMALL, TRAIN_FULL)

	SPLIT_SIZE = 1000
	TEST_FULL:str = 'TweetsFullTest.csv'
	RunTweetsSentimentLarge(TEXT_COL, SENTIMENT_COL, TRAIN_FULL, TEST_FULL, SPLIT_SIZE)

	PREDICT_FILE:str = 'ItaiGOT_7_B_before_SA-482000.excel.csv'
#	RunTweetsSentimentLarge(TEXT_COL, SENTIMENT_COL, TRAIN_FULL, PREDICT_FILE, SPLIT_SIZE)

	return

if __name__ == '__main__':
	main(sys.argv)