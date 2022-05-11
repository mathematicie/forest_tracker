import tkinter as tk
from tkinter.filedialog import askopenfilename
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

window = tk.Tk()
window.withdraw()
filename = askopenfilename()


model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open(filename)

size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(str(int(prediction[0][0]*100))+ "% Tree")
print(str(int(prediction[0][1]*100))+ "% Not Tree")
