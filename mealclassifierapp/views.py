import os

import pandas as pd
from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from meal_classifier2.settings import BASE_DIR



def train_model():

    file_path = os.path.join(BASE_DIR, 'mealclassifierapp', 'data', 'meal_data.csv')
    data = pd.read_csv(file_path)
    descriptions = data['description']
    labels = data['category']


    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(descriptions)


    model = MultinomialNB()
    model.fit(X, labels)

    return model, vectorizer


model, vectorizer = train_model()


def form_page(request):
    github_repo_url =  "https://github.com/jmoncla/meal_classifier.git"
    return render(request, 'mealclassifierapp/form_page.html', {'github_repo_url': github_repo_url})


def predict_meal(request):
    prediction = None
    if request.method == 'POST':

        user_input = request.POST.get('meal_input', '')

        if user_input:

            input_vector = vectorizer.transform([user_input])

            prediction = model.predict(input_vector)[0]

    return render(request, 'mealclassifierapp/result_page.html', {'prediction': prediction})
