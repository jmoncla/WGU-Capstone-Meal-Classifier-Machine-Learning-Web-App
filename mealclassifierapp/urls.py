from django.urls import path
from . import views

urlpatterns = [
        path('', views.form_page, name='form_page'),  # Landing page for form
        path('predict/', views.predict_meal, name='predict_meal'),  # Results page
]
