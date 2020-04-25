from PIL import Image, ImageDraw, ImageFont
import numpy as np

im = 'static/data/recognized_faces/group-photo-ideas_1024x1024.jpg'

image = Image.open(im).convert('RGB')

draw = ImageDraw.Draw(image)


w, h = image.size

font_size = int(np.mean([w, h]) * 0.04)
print(font_size)

font = ImageFont.truetype(
    "static/assets/fonts/Roboto-Regular.ttf", font_size)

draw.text((100, 100), "Animikh Aich", fill='red', font=font)

image.show()
