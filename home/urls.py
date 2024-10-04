from django.contrib import admin
from django.urls import path
from home import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path("", views.index, name='home'),
    path("About/", views.About, name='About'),
    path("Articles/", views.Articles, name='Articles'),
    path('detect_ai/', views.detect_ai, name='detect_ai'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)