import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from decouple import config
import os
import uuid

class OCRView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No image uploaded'}, status=400)

        # Save to a temporary file
        temp_filename = f'/tmp/{uuid.uuid4().hex}.jpg'
        with open(temp_filename, 'wb') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Call Hugging Face Space API
        hf_space_url = config('HF_SPACE_URL', default='https://YOUR_USER-amharic-ocr.hf.space/api/predict')
        with open(temp_filename, 'rb') as f:
            files = {'data': (temp_filename, f, 'image/jpeg')}
            response = requests.post(hf_space_url, files=files)

        os.unlink(temp_filename)

        if response.status_code != 200:
            return Response({'error': 'OCR service failed'}, status=500)

        result = response.json()
        # Gradio returns: {'data': ['recognized text']}
        text = result.get('data', [''])[0] if isinstance(result, dict) else ''
        return Response({'text': text})
