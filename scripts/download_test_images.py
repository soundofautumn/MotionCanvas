import os, requests
from PIL import Image

OUT = "data/input_img"
os.makedirs(OUT, exist_ok=True)

# Unsplash 上找的一些简单主体图片（经缩放和 jpg 转 png）
urls = {
    "dog.png":  "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=640",      # dog close-up
    "bird.png": "https://images.unsplash.com/photo-1552728089-57bdde30beb3?w=640",      # bird
    "car.png":  "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=640",   # car
}

for name, url in urls.items():
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        print(f"Skip {name} (exists)")
        continue
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    img.save(path, "PNG")
    print(f"Saved {name} ({img.size})")
