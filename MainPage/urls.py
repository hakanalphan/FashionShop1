from django.urls import path
from django.urls import path
from . import views



from MainPage import views
urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.fashion_recommender, name='upload'),

]