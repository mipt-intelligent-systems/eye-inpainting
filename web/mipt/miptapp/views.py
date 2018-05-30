import os
import json
from scipy import misc

from django.http import HttpResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from images_tools import *




def index(request):
    return HttpResponse("Hello, world.")


def handle_uploaded_file(image):
    image_name = 'images/input/' + image.name
    location = os.getcwd()
    print(location)
    with open(image_name, 'wb+') as destination:
        for chunk in image.chunks():
            destination.write(chunk)
    return image_name


@method_decorator(csrf_exempt, name='dispatch')
class FileLoader(View):
    @csrf_exempt
    def get(self, request):
        return HttpResponse(status=200)

    @csrf_exempt
    def post(self, request):
        temp_image_file = request.FILES['img']
        input_image_name = handle_uploaded_file(temp_image_file)
        image = misc.imread(input_image_name)
        result_image = simple_function(image)
        output_image_name = 'images/output/' + input_image_name.split('/')[-1]
        misc.imsave(output_image_name, result_image)

        r = {
            'status': 'success',
            'input_image': input_image_name,
            'output_image': output_image_name
        }
        response = HttpResponse(json.dumps(r), content_type="application/json")
        response['Access-Control-Allow-Origin'] = '*'
        return response
