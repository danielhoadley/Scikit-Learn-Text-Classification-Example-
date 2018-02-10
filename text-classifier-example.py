import sklearn
import numpy as np
from glob import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline

def pathfinder(targetPath):
    path_string = targetPath.replace('/path/to/project/folder/','')
    path_string = path_string.strip('/')
    return path_string

# Get paths to labelled data
rawFolderPaths = glob("/path/to/project/folder/*/")

print ('\nGathering labelled categories...\n')

categories = []

# Extract the folder paths, reduce down to the label and append to the categories list
for i in rawFolderPaths:
    category = pathfinder(i)
    categories.append(category)

# Load the data
print ('\nLoading the dataset...\n')
docs_to_train = sklearn.datasets.load_files("/Users/danielhoadley/PycharmProjects/trainer/!labelled_data_reportXML",
                                            description=None, categories=categories, load_content=True,
                                            encoding='utf-8', shuffle=True, random_state=42)

# Split the dataset into training and testing sets
print ('\nBuilding out hold-out test sample...\n')
X_train, X_test, y_train, y_test = train_test_split(docs_to_train.data, docs_to_train.target, test_size=0.4)


# Construct the classifier pipeline using a SGDClassifier algorithm
print ('\nApplying the classifier...\n')
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42, verbose=1)),
])

# Fit the model to the training data
text_clf.fit(X_train, y_train)

# Run the test data into the model
predicted = text_clf.predict(X_test)

# Calculate mean accuracy of predictions
print (np.mean(predicted == y_test))

# Generate labelled performance metrics
print(metrics.classification_report(y_test, predicted,
    target_names=docs_to_train.target_names))