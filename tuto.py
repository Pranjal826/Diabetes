import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
# Load the dataset
df = pd.read_csv('Fake.csv')

# Split into features and target
X = df['text']
y = df['title']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training set
X_train = vectorizer.fit_transform(X_train)

# Transform the testing set
X_test = vectorizer.transform(X_test)
# Initialize the classifier
clf = LogisticRegression()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred = clf.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate the confusion matrix of the model
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(cm)
