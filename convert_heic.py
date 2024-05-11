import PIL.Image 
import os

def convert_heic_to_jpg(heic_file_no,num_img):
    for i in range(num_img):
        heic_file = 'img/statue/IMG_'+str(heic_file_no+i)+'.HEIC'
        # heic_file = 'img/statue/IMG_2749.HEIC'
        with PIL.Image.open(heic_file) as img:
            jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
            img.convert("RGB").save(jpg_file, "JPEG")
            print(f"{heic_file} converted to {jpg_file}")

convert_heic_to_jpg(2749,30)