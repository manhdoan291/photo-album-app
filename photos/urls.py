from django.urls import path
from . import views

urlpatterns = [
    path('photo/<str:pk>/', views.extract_info, name='extract_info'),
    path('', views.gallery, name='gallery'),
    path('photo/<str:pk>/', views.viewPhoto, name='photo'),
    path('add/', views.addPhoto, name='add')
]
