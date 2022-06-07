"""covid19 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from covid19 import views
from covid19 import apis

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.render_home_page),
    path('search/', views.render_query_results),
    path('quote/', views.render_quote),
    path('about/', views.render_about),

    # APIs
    path('api/search', views.api_search),
    path('api/tags', views.api_get_all_tags)
]
