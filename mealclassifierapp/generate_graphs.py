import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

from meal_classifier2.settings import BASE_DIR
from views import train_model

# Load the data
file_path = os.path.join(BASE_DIR, 'mealclassifierapp', 'data', 'testmodeldata2.csv')
data = pd.read_csv(file_path)  # Adjust path as needed

# Descriptive Method: Calculate accuracy (Example usage with testing split)
def calculate_accuracy(model, vectorizer):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    descriptions = data['description']
    labels = data['category']
    X = vectorizer.transform(descriptions)
    y_pred = model.predict(X)
    return accuracy_score(labels, y_pred)

# Visual: Word Usage Chart by Category
def generate_word_usage_chart():
    vectorizer = CountVectorizer(max_features=10)
    X = vectorizer.fit_transform(data['description'])
    word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    word_counts['category'] = data['category']
    top_words = word_counts.groupby('category').sum().T

    top_words.plot(kind='bar', figsize=(10, 6), title='Top Words by Category')
    plt.ylabel('Frequency')
    plt.savefig('word_usage_chart.png')

# Visual: Average Words Used by Category
def generate_avg_words_chart():
    data['word_count'] = data['description'].apply(lambda x: len(x.split()))
    avg_words = data.groupby('category')['word_count'].mean()

    avg_words.plot(kind='bar', color='skyblue', figsize=(8, 5), title='Average Words by Category')
    plt.ylabel('Average Word Count')
    plt.savefig('avg_words_chart.png')

# Visual: Dataset Distribution
def generate_dataset_distribution_chart():
    data['category'].value_counts().plot(kind='bar', color='coral', figsize=(8, 5), title='Dataset Distribution by Category')
    plt.ylabel('Number of Entries')
    plt.savefig('dataset_distribution_chart.png')

# Generate all visuals
def generate_all_visuals():
    generate_word_usage_chart()
    generate_avg_words_chart()
    generate_dataset_distribution_chart()

# Run this script to generate visuals
if __name__ == '__main__':
    model, vectorizer = train_model()  # Train model for accuracy
    accuracy = calculate_accuracy(model, vectorizer)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

   # generate_all_visuals()
   # print("Visuals generated and saved.")