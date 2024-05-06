from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from MainPage import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('MainPage.urls')),
    #path('upload/', views.fashion_recommender, name='upload'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

