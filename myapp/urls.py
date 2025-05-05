from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home', views.home, name='home'),
    path('home_2', views.home_2, name='home_2'),
    path('home_3', views.home_3, name='home_3'),
    path('dashboard', views.dashboard, name='dashboard'),
]