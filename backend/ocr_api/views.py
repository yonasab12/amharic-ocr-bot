from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
import requests
import os
from decouple import config

class OCRView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No image provided'}, status=400)

        # Save temporarily
        temp_path = '/tmp/uploaded.jpg'
        with open(temp_path, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Call Hugging Face Inference API
        api_key = config('HF_API_KEY')
        model_id = 'your-username/your-ocr-model'  # or use a public one
        headers = {'Authorization': f'Bearer {api_key}'}
        with open(temp_path, 'rb') as f:
            response = requests.post(
                f'https://api-inference.huggingface.co/models/{model_id}',
                headers=headers,
                data=f.read()
            )
        if response.status_code != 200:
            return Response({'error': 'Model inference failed'}, status=500)

        result = response.json()
        # Assuming model returns {"text": "..."}
        text = result.get('text', '')
        return Response({'text': text})
