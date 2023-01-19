from fastai.vision.all import *
import gradio as gr

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

#!export
pollutant_labels = (
    "antennas",
    "billboard",
    "broken roads",
    "construction sites",
    "electric pole",
    "garbage can",
    "graffiti",
    "smog",
    "street litter"
)

model = load_learner('vispol-1-recognizer-v0.pkl')

def recognize_image(image):
  pred, idx, probs = model.predict(image)
  return dict(zip(pollutant_labels, map(float, probs)))

#!export
image = gr.inputs.Image(shape=(256,256))
label = gr.outputs.Label(num_top_classes=5)
examples = [
    'unknown_00.jpg',
    'unknown_01.jpg',
    'unknown_02.jpg',
    'unknown_03.jpg',
    'unknown_04.jpg'
    ]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)