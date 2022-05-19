# import all the modules
import string
import numpy as np
import pickle
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")

# read the dataset file
df = pd.read_csv('spam-filtering-ml-code/spam.csv')
message_X = df.iloc[:, 1]  # EmailText
labels_Y = df.iloc[:, 0]  # Label

# stemming variable initialization
lstem = LancasterStemmer()


def mess(messages):
    message_x = []
    for me_x in messages:

        # filter out other datas except alphabets
        me_x = ''.join([char for char in str(
            me_x) if char not in string.punctuation])
        # tokenize or split the messages into respective words
        words = word_tokenize(me_x)
        # stem the words to their root words
        message_x += [' '.join([lstem.stem(word) for word in words])]
    return message_x


message_x = mess(message_X)
# vectorization process
# ignore stop words i.e. words that are of least importance
tfvec = TfidfVectorizer(stop_words='english')
# vectorize feature data
x_new = tfvec.fit_transform(message_x).toarray()

# replace ham and spam with 0 and 1 respectively
y_new = np.array(labels_Y.replace(to_replace=['ham', 'spam'], value=[0, 1]))

# split our dataset into training and testing part
x_train, x_test, y_train, y_test = ttsplit(
    x_new, y_new, test_size=0.2, shuffle=True)
# use svm classifier to fit our model for training process
classifier = svm.SVC()
classifier.fit(x_train, y_train)

# store the classifier as well as messages feature for prediction
pickle.dump({'classifier': classifier, 'message_x': message_x},
            open("training_data.pkl", "wb"))
print('done')
