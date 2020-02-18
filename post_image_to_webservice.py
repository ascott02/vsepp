import requests
import os
# https://www.techcoil.com/blog/how-to-upload-a-file-and-some-data-through-http-multipart-in-python-3-using-the-requests-library/ 

filename = "./COCO_val2014_000000391895.jpg"
caption = "A man wearing a red jacket riding a motor bike on a dirt road on the countryside."
page = 'http://localhost:8080/api'
token = ''

def send_data_to_server(image_path, caption, page):
    image_filename = os.path.basename(filename)
    multipart_form_data = {
        'token': ('', str(token)),
        'caption': ('', str(caption)),
        'img_file': (image_filename, open(image_path, 'rb')),
    }
 
    response = requests.post(page, files=multipart_form_data)
    print(response.status_code, response.text)

send_data_to_server(filename, caption, page)


