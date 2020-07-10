from __future__ import print_function

# for web.py
import web
import re
import config
import base64

# for vsepp and stand alone caption and image cosine similarity calculation
import os
import sys
import torch
import pickle

sys.path.append('../')
from vocab import Vocabulary
# from data import get_test_loader
import data
from model import VSE
from evaluation import i2t, t2i, encode_data, evalrank

import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
import nltk

import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import io
import requests

# web.py
render = web.template.render('templates/', cache=config.cache)

# web.py
urls = (
    '/', 'index',
    '/api', 'api',
    '/login', 'login',
    '/upload', 'upload',
)

# vsepp
model_path = "../runs/coco_vse++_resnet_restval/model_best.pth.tar" # pretrained image model
data_path="../data/" 
data_name="coco"
vocab_path="../vocab/"
run_path="../runs/"
split = "test" # use the test split for analysis

# load the model from the saved file and get the opt
checkpoint = torch.load(model_path)  
opt = checkpoint['opt']

# set the data_path into the opt, in case it had changed
opt.data_path = data_path
opt.data_name = data_name
    
# opt.vocab_path is relative to last run too
opt.vocab_path = vocab_path
opt.run_path = run_path

# load vocabulary used by the model
with open(os.path.join(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb') as f:
    vocab = pickle.load(f)

# save the vocab size to the opt
opt.vocab_size = len(vocab)

# opt.workers = 4

web.debug("opt:", vars(opt))

model = VSE(opt)

# load model state
model.load_state_dict(checkpoint['model'])

# don't forget to enable eval mode
model.val_start()

def _get_score(img, cap):
    transform = data.get_transform(data_name,split, opt)      
    
    if img.startswith('http'):
        path = requests.get(img, stream=True)
        try:
            print("DEBUG try")
            img = Image.open(path).convert('RGB')
        except:
            print("DEBUG except")
            stream = io.BytesIO(path.content)
            img = Image.open(stream).convert('RGB')
    else:
        # path = image_path
        img = Image.open(io.BytesIO(img)).convert('RGB')

    img_tens = transform(img).unsqueeze(0)
    if torch.cuda.is_available():
       img_tens = img_tens.cuda()

    img_emb = model.img_enc(img_tens)
    img_emb = img_emb.cuda().detach().cpu().clone().numpy()
    
    new_cap_tokens = nltk.tokenize.word_tokenize(str(cap).lower())
    new_cap_lst = []
    new_cap_lst.append(vocab("<start>"))
    new_cap_lst.extend([vocab(token) for token in new_cap_tokens])
    new_cap_lst.append(vocab("<end>"))

    new_cap = torch.Tensor([new_cap_lst]).long()
    if torch.cuda.is_available():
       new_cap = new_cap.cuda()

    lengths = torch.from_numpy(np.array([len(new_cap_lst)]))
    new_cap_emb = model.txt_enc.forward(new_cap, lengths)
    new_cap_emb = new_cap_emb.cuda().detach().cpu().clone().numpy()
    captions = new_cap_emb

    img_emb = img_emb.astype('float64')
    captions = captions.astype('float64')

    web.debug("DEBUG img_emb: ", type(img_emb), img_emb.dtype, img_emb.shape, len(img_emb))
    web.debug("DEBUG captions: ", type(captions), captions.dtype, captions.shape, len(captions))

    d = np.dot(img_emb, captions.T).flatten()
    inds = np.argsort(d)[::-1]
    web.debug("DEBUG d, inds: ", d, inds)
    return d[inds[0]]

class index:

    def GET(self, *args):

        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return """<html><head></head><body>
This form takes an image upload and caption description and returns cosine-similarity.<br/><br/>
<form method="POST" enctype="multipart/form-data" action="">
Image: <input type="file" name="img_file" /><br/><br/>
Caption: <input type="text" name="caption" />
<br/><br/>
<input type="submit" />
</form>
</body></html>"""
        else:
            raise web.seeother('/login')

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['caption'])              # This is the caption contents
        web.debug(x['img_file'].filename)    # This is the filename
        data_uri = base64.b64encode(x['img_file'].file.read())
        img_tag = '<img src="data:image/jpeg;base64,{0}">'.format(data_uri)

        score = _get_score(x['img_file'].value, str(x['caption']))

        page = """<html><head></head><body>
This form takes an image upload and caption description and returns cosine-similarity.<br/><br/>
<form method="POST" enctype="multipart/form-data" action="">
Image: <input type="file" name="img_file" /><br/><br/>
Caption: <input type="text" name="caption" /><br/>
<br/><br/>
<input type="submit" />
</form>""" + img_tag + """<br/>Caption: """ + str(x['caption']) + """<br/>
Score: """ + str(score) + """ 
</body></html>"""

        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return page
        else:
            raise web.seeother('/login')


class api:

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['token'])                  # This is the api token 
        web.debug(x['caption'])                # This is the caption contents
        web.debug(x['img_url'])              # This is the URL to the image

        if not x['caption']:
            return "No caption."

        if not x['img_url']:
            return "No file."

        if not x['token']:
            return "No token."

        if not x['token'] in config.tokens:
            return "Not in tokens."
    
        score = _get_score(str(x['img_url']), str(x['caption']))
        return score


class upload:

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['token'])                  # This is the api token 
        web.debug(x['caption'])                # This is the caption contents
        web.debug(x['img_file'].filename)      # This is the filename
        # web.debug(x['img_file'].value)       # This is the file contents
        # web.debug(x['img_file'].file.read()) # Or use a file(-like) object
        # data = web.data()

        if not x['caption']:
            return "No caption."

        if not x['img_file'].filename:
            return "No file."

        if not x['token']:
            return "No token."

        if not x['token'] in config.tokens:
            return "Not in tokens."
    
        data_uri = base64.b64encode(x['img_file'].file.read())
        img_tag = '<img src="data:image/jpeg;base64,{0}">'.format(data_uri)

        score = _get_score(x['img_file'].value, str(x['caption']))

        return score


class login:

    def GET(self):
        auth = web.ctx.env.get('HTTP_AUTHORIZATION')
        authreq = False
        if auth is None:
            authreq = True
        else:
            auth = re.sub('^Basic ','',auth)
            username,password = base64.decodestring(auth).split(':')
            if (username,password) in config.allowed:
                raise web.seeother('/')
            else:
                authreq = True
        if authreq:
            web.header('WWW-Authenticate','Basic realm="Auth example"')
            web.ctx.status = '401 Unauthorized'
            return


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()


