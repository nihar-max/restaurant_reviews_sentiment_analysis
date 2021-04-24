# Importing essential libraries
import pandas as pd
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer    # lemmatize
import re
# Loading the dataset
df = pd.read_csv('Restaurant_Reviews.txt', delimiter='\t', quoting=3)
# Cleaning the reviews
corpus = []
for i in range(0,1000):

  # Cleaning special character from the reviews
  review = re.sub(pattern='[^a-zA-Z]',repl=' ', string=df['Review'][i])

  # Converting the entire review into lower case
  review = review.lower()

  # Tokenizing the review by words
  review_words = review.split()

  # Removing the stop words
  review_words = [word for word in review_words if not word in set(stopwords.words('english'))]

  # Stemming the words
  lemmatizer = WordNetLemmatizer()
  review = [lemmatizer.lemmatize(word) for word in review_words]

  # Joining the stemmed words
  review = ' '.join(review)

  # Creating a corpus
  corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(max_features=1000)
X = tf_idf.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values
# Creating a pickle file for the Tf-idf
pickle.dump(tf_idf, open('tf-transform.pkl', 'wb'))


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Selecting best model on this data Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression(C=2,solver = "liblinear",penalty='l1',multi_class='ovr')
#For small datasets, ‘liblinear’ is a good choice
#‘liblinear’ and ‘saga’ also handle L1 penalty
clf_log.fit(X_train,y_train)

# Creating a pickle file for the Logistic model
filename = 'restaurant-sentiment-logistic-model.pkl'
pickle.dump(clf_log, open(filename, 'wb'))