from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='chatbot'),
    path('handleInput/', views.handleInput, name='handleInput'),
]