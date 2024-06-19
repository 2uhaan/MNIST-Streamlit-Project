import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np

model_new = keras.models.load_model('/content/mnist.hdf5')

st.title("MNIST Digit Recognizer")

SIZE = 192

canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=10,
    stroke_color='#ffffff',
    background_color="#000000",
    height=150, width=150,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (SIZE, SIZE))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img_rescaled = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_NEAREST)
    img_input = np.expand_dims(img_rescaled, axis=-1)  # Add channel dimension
    st.write('Input Image')
    st.image(img_rescaled, clamp=True)

if st.button('Predict') and canvas_result.image_data is not None:
    pred = model_new.predict(np.array([img_input]))
    st.write(f'Result: {np.argmax(pred[0])}')
    st.bar_chart(pred[0])

