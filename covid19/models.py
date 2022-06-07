from django.db import models

class SearchCache(models.Model):
    query_text = models.TextField()
    query_response = models.TextField()
    query_time = models.DateTimeField()