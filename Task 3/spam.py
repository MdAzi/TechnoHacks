import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

file_path = ('C:/Users/ELCOT/Desktop/Task 3/spam.csv')
data = pd.read_csv(file_path, encoding='ISO-8859-1')

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(data['message'])

clf = MultinomialNB()
clf.fit(X_counts, data['label'])

data['predicted_label'] = clf.predict(X_counts)

spam_emails = data[data['predicted_label'] == 'spam']

print("Spam Emails:")
print(spam_emails[['message']])

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.3, random_state=42)

X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

clf.fit(X_train_counts, y_train)

y_pred = clf.predict(X_test_counts)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
