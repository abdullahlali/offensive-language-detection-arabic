ArabGuard: Arabic Offensive Language Detection System
Project Overview
ArabGuard is an Arabic offensive language detection system designed to identify offensive or inappropriate comments in Arabic text. The system uses machine learning techniques to analyze the input text and classify it as either offensive or non-offensive. By using the SVC (Support Vector Classifier) model with a TF-IDF vectorizer, ArabGuard provides an accurate way to filter out harmful content, ensuring a safer online environment, especially on social media platforms.

This project aims to help society by making social media and other online platforms safer by detecting and filtering curse words, hate speech, or any offensive language in Arabic. The system uses state-of-the-art techniques and is designed to work with Arabic text, ensuring it can help moderate content on platforms that have a large Arabic-speaking audience.

Model Performance
Accuracy: 84.12%
Approach: The model uses TF-IDF vectorization and a Support Vector Classifier (SVC) with balanced class weights to improve performance on imbalanced data.
Objective: The goal is to classify Arabic text as either offensive or non-offensive, making the system suitable for use in content moderation.
Dependencies
To run this project, you will need the following dependencies:

pandas – For data manipulation and handling datasets.
nltk – For natural language processing tasks like tokenization and stopword removal.
scikit-learn – For machine learning algorithms, model training, and text vectorization.
openpyxl – To read Excel files.
re – For regular expressions in text preprocessing.
You can install the necessary dependencies by running the following command:

bash
Copy code
pip install pandas nltk scikit-learn openpyxl
Project Structure
bash
Copy code
arab_guard/
│
├── arabic_offensive_dataset.xlsx  # The dataset containing the Arabic text and labels
├── arab_guard.py                  # Main Python script
├── README.md                      # This README file
└── requirements.txt               # The dependencies list
How to Use
Prepare the Dataset: Ensure that you have the arabic_offensive_dataset.xlsx file with the following structure:

Comment: The Arabic text to be analyzed.
Majority_Label: The label (Offensive or Non-Offensive) for each comment.
Run the Script: To run the system, use the following command:

bash
Copy code
python arab_guard.py
Input a Comment: After running the script, the system will prompt you to enter a comment for analysis. For example:

css
Copy code
Enter a comment to analyze: @User.IDX هذا خكري مايستاهل حتى الاستماع له !!!
Output: The system will output whether the input text is offensive or not, along with the confidence score:

vbnet
Copy code
Your text is: Offensive
Confidence: 0.939
Continue or Exit: You can continue analyzing other texts or exit the program.

How It Works
1. Preprocessing
The input text is preprocessed to remove non-alphabetic characters, diacritics (Arabic vocalization marks), and stopwords.
The text is tokenized, and only relevant words are kept for analysis.
2. Vectorization
The text is converted into a numerical format using TF-IDF vectorization. This method transforms the text into vectors based on the importance of each word in the context of the dataset.
3. Model Training
The system uses a Support Vector Classifier (SVC) to train the model on the processed and vectorized dataset. The class weights are balanced to address any class imbalance, improving model performance.
4. Prediction
Once trained, the model can predict whether new input text is offensive or not. The model also provides a confidence score indicating the likelihood of the prediction.
How It Helps Society and Social Media
ArabGuard helps create a safer online environment by:

Detecting Offensive Language: It identifies and classifies offensive comments or hate speech in Arabic text, helping content moderators quickly remove harmful content.
Improving Social Media Safety: The system ensures that online platforms remain a safe and welcoming space for everyone by filtering out offensive language.
Automating Content Moderation: By automating the detection of offensive language, the system reduces the manual effort required for content moderation and improves the speed at which harmful content is flagged.
Promoting Positive Online Communication: ArabGuard helps promote more respectful and constructive conversations by detecting and removing inappropriate language.
Future Improvements
Expand to Multilingual Support: Although the current system is designed for Arabic, it can be expanded to other languages with similar techniques.
Use Deep Learning Models: Implementing deep learning models such as BERT for Arabic text can improve accuracy further.
Incorporate Real-time Analysis: Modify the system to allow real-time analysis for live discussions and comment sections.
