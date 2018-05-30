from django.urls import path

from .views import *

app_name = 'miptapp'

urlpatterns = [
    path('', index, name='index'),
    path('upload/', FileLoader.as_view(), name='upload')
]
