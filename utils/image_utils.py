import os
import cv2
import rembg
import numpy as np


def preprocess(img_path, img_size=256):
    border_ratio = 0.2
    recenter = True
    session = rembg.new_session(model_name="u2net")

    out_base = os.path.basename(img_path).split('.')[0]
    out_rgba = os.path.join('/src/' + out_base + '_rgba.png')
    
    # load image
    print(f'[INFO] loading image {img_path}...')
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # Carve background
    print(f'[INFO] background removal...')
    # RGBA image of shape [height, width, 4]
    carved_image = rembg.remove(image, session=session) 
    mask = carved_image[..., -1] > 0

    # Recenter image
    if recenter:
        print(f'[INFO] recenter...')
        final_rgba = np.zeros((img_size, img_size, 4), dtype=np.uint8)
        
        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        height = x_max - x_min
        width = y_max - y_min
        desired_size = int(img_size * (1 - border_ratio))
        scale = desired_size / max(height, width)

        height_new = int(height * scale)
        width_new = int(width * scale)
        x2_min = (img_size - height_new) // 2
        x2_max = x2_min + height_new
        y2_min = (img_size - width_new) // 2
        y2_max = y2_min + width_new
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            carved_image[x_min:x_max, y_min:y_max], 
            (width_new, height_new), 
            interpolation=cv2.INTER_AREA
        )
    else:
        final_rgba = carved_image
    
    # write image
    cv2.imwrite(out_rgba, final_rgba)
    return out_rgba