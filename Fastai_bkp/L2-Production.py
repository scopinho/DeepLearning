#from fastcore.all import *
#from fastdownload import download_url
from fastai.vision.all import *
#from time import sleep
import gradio as gr
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def guess_bear(x): return x[0].isupper()

learn = load_learner('class_bears.pkl')

categories = ['black', 'grizzly', 'teddy']

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['bear_black.jpg', 'bear_grizzly.jpg', 'bear_teddy.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)