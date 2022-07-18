import os
import time
import numpy as np
import cv2
import glob
import tensorflow as tf
import tensorflow_hub as hub
import DOW_AI_upscale_functions as DAU
import PIL

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

# Convert all images from tga to png for processing by esrgan
for file in glob.glob("images_original/*.tga"):
    DAU.TGAtoPNG(file)

# Upscaling of images
temp_path = "images_temp/*.png"
for file in glob.glob(temp_path):
    savename = "images_upscaled/" + os.path.basename(file.replace('.png', ''))
    IMAGE_PATH = file

    if DAU.texture_identify(file) == "Texture_Default":
        # Process default texture separate as it requires an Alpha channel
        hr_image = DAU.preprocess_image_default(IMAGE_PATH)
        # DAU.plot_image(tf.squeeze(hr_image), title="Original Image")
        model = hub.load(MODEL_PATH)
        start = time.time()
        fake_image = model(hr_image)
        fake_image = tf.squeeze(fake_image)
        print("Time Taken: %f" % (time.time() - start))

        # DAU.plot_image(tf.squeeze(fake_image), title="Super Resolution")
        UPSCALE_ALPHA = cv2.cvtColor(np.asarray(tf.squeeze(fake_image)), cv2.COLOR_RGB2RGBA)
        UPSCALE_ALPHA[:, :, 3] = DAU.ScaledAlphaChannelArray(IMAGE_PATH)

        print(str(DAU.texture_identify(file)) +
              " should be shape (2048, 2048, 4), is actually: " + str(UPSCALE_ALPHA.shape))

        DAU.save_image(UPSCALE_ALPHA, filename=savename)
        # DAU.plot_image(UPSCALE_ALPHA)

    elif DAU.texture_identify(file) == "Texture_Badge":
        DAU.Generate_Badge_Layer(file, savename)
        print("Badge layer generated")

    elif DAU.texture_identify(file) == "Texture_Dirt" \
            or "Texture_Eyes"\
            or "Texture_Primary" \
            or "Texture_Secondary" \
            or "Texture_Trim" \
            or "Texture_Weapon":

        # Process other textures separate as they are 8-bit grayscale images
        hr_image = DAU.preprocess_image_other(IMAGE_PATH)
        # DAU.plot_image(tf.squeeze(hr_image), title="Original Image")
        model = hub.load(MODEL_PATH)
        start = time.time()
        fake_image = model(hr_image)
        fake_image = tf.squeeze(fake_image)
        print("Time Taken: %f" % (time.time() - start))
        # DAU.plot_image(tf.squeeze(fake_image), title="Super Resolution")

        fake_image = cv2.cvtColor(np.asarray(fake_image), cv2.COLOR_RGB2GRAY)
        DAU.save_image(tf.squeeze(fake_image), filename=savename)
        # DAU.plot_image(tf.squeeze(fake_image), title="Super Resolution")

        print(str(DAU.texture_identify(file)) + " should be shape (2048, 2048), is actually: " + str(fake_image.shape))
    else:
        print("Cannot identify texture... Skipping file:" + str(file))


for file in glob.glob(temp_path):
    os.remove(file)
    print("Cleaning temp folder, deleting... " + file)
