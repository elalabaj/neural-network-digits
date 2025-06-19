import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np

class DigitRecognizer:
    def __init__(self, model):
        self.model = model

        self.window = tk.Tk()
        self.window.title("Digit recognition")
        self.canvas = tk.Canvas(self.window, width=280, height=280, bg='white')
        self.canvas.pack()

        self.image = Image.new('L', (280,  280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind('<B1-Motion>', self.draw_digit)

        tk.Button(self.window, text='Predict', command=self.predict).pack()
        tk.Button(self.window, text='Clear', command=self.clear).pack()

    def draw_digit(self, event):
        x, y = event.x, event.y
        r = 7.5
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear(self):
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, 280, 280], fill=255)

    def predict(self):
        img_resized = self.image.resize((28, 28))
        img_inverted = ImageOps.invert(img_resized)
        img_array = np.array(img_inverted) / 255.0
        img_array = img_array.reshape((28*28, 1))

        prediction = self.model.feedforward(img_array)[0][-1]
        predicted_digit = np.argmax(prediction)
        certainty = prediction[predicted_digit][0]
        print(f'Predicted {predicted_digit} with {(certainty*100):.2f}% certainty')

    def run(self):
        self.window.mainloop()