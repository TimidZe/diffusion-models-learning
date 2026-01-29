from PIL import Image

def process_mobile_photo(image_path, output_size=(512, 512)):
    img = Image.open(image_path)
    w, h = img.size

    short_edge = min(w, h)
    left = (w - short_edge) / 2
    top = (h - short_edge) / 2
    right = (w + short_edge) / 2
    bottom = (h + short_edge) / 2

    img_cropped = img.crop((left, top, right, bottom))

    img_resized = img_cropped.resize(output_size, Image.LANCZOS)

    return img_resized

processed_img = process_mobile_photo("mycat.jpg", output_size=(512, 512))
processed_img.save("mycat_512.jpg")