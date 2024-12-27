import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from meal_classifier2.settings import BASE_DIR
from views import train_model


test_file_path = os.path.join(BASE_DIR, 'mealclassifierapp', 'data', 'testmodeldata2.csv')
test_data = pd.read_csv(test_file_path)


meal_file_path = os.path.join(BASE_DIR, 'mealclassifierapp', 'data', 'meal_data.csv')
meal_data = pd.read_csv(meal_file_path)


def calculate_accuracy(model, vectorizer):
    from sklearn.model_selection import train_test_split

    descriptions = test_data['description']
    labels = test_data['category']
    X = vectorizer.transform(descriptions)
    y_pred = model.predict(X)
    return accuracy_score(labels, y_pred), y_pred

def generate_word_usage_chart():
    vectorizer = CountVectorizer(stop_words='english', max_features=20)  # Exclude common stop words
    X = vectorizer.fit_transform(meal_data['description'])  # Transform the descriptions into a feature matrix
    word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    word_counts['category'] = meal_data['category']
    category_word_counts = word_counts.groupby('category').sum()  # Sum word counts per category


    ax = category_word_counts.T.plot(kind='bar', figsize=(12, 8), stacked=False)
    ax.set_title('Top Words by Category')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Words')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('word_usage_chart.png')
    plt.close()


def generate_avg_words_line_chart():
    meal_data['word_count'] = meal_data['description'].apply(lambda x: len(x.split()))
    avg_words = meal_data.groupby('category')['word_count'].mean()

    avg_words.plot(kind='line', color='skyblue', figsize=(8, 5), title='Average Words by Category')
    plt.ylabel('Average Word Count')
    plt.savefig('avg_words_line_chart.png')
    plt.close()

# Pie chart showing category distribution
def generate_dataset_distribution_chart():
    category_counts = meal_data['category'].value_counts()


    ax = category_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), startangle=90)
    ax.set_title('Dataset Distribution by Category')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('dataset_distribution_chart.png')
    plt.close()


def generate_confusion_matrix(model, vectorizer):
    descriptions = meal_data['description']
    labels = meal_data['category']
    X = vectorizer.transform(descriptions)
    y_pred = model.predict(X)

    cm = confusion_matrix(labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()



def generate_all_visuals(model, vectorizer):
    generate_word_usage_chart()
    generate_avg_words_line_chart()
    generate_dataset_distribution_chart()
    generate_confusion_matrix(model, vectorizer)


if __name__ == '__main__':
    model, vectorizer = train_model()
    accuracy, y_pred = calculate_accuracy(model, vectorizer)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    #generate_all_visuals(model, vectorizer)
    #print("Visuals generated and saved.")
