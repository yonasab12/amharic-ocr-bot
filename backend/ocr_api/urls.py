from django.urls import path
from .views import OCRView

urlpatterns = [
    path('predict', OCRView.as_view(), name='ocr'),
]