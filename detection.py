import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the dataset
file_path = 'arabic_offensive_dataset.xlsx'
df = pd.read_excel(file_path)

# Download Arabic stopwords
nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))

def preprocess_arabic_text(text):
    # Remove diacritics and non-alphabetic characters
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\u064B-\u0652]', '', text)  # Arabic diacritics
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in arabic_stopwords]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['Comment'].apply(preprocess_arabic_text)

# Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])

# Labels
y = df['Majority_Label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC(probability=True, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Model accuracy: {accuracy * 100:.2f}%")

while True:
    # Input from user
    user_input = input("Enter a comment to analyze: ")
    print(f"Input as received: {user_input}")

    # Preprocess the input
    cleaned_input = preprocess_arabic_text(user_input)

    # Transform the input using the saved vectorizer
    input_vector = vectorizer.transform([cleaned_input])

    # Predict the label
    prediction = model.predict(input_vector)[0]
    # Predict confidence scores (probabilities)
    confidence_scores = model.predict_proba(input_vector)[0]
    max_specific_confidence = max(confidence_scores)

    # Output results
    print(f"Your text is: {prediction}")
    print(f"Confidence: {max_specific_confidence}\n")
    
    follow = input("Would you like to continue analyzing other texts? (yes/no): ").lower()

    if follow == "no":
        break