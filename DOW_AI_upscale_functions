import os
import PIL.Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

#Define functions
def TGAtoPNG(Image_path):
    filename, file_extension = os.path.splitext(Image_path)
    filename = os.path.basename(filename)
    PIL.Image.open(Image_path).save("images_temp/" + filename + ".png")
    return "images_temp/" + filename + ".png"

def preprocess_image_default(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def preprocess_image_other(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = np.asarray(PIL.Image.open(image_path))
  hr_image = np.repeat(hr_image[:, :, np.newaxis], 3, axis=2)
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save.
  """
  if not isinstance(image, PIL.Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = PIL.Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.tga" % filename)

def plot_image(image, title=""):
  """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = PIL.Image.fromarray(tf.cast(image, tf.uint8).numpy())
  plt.imshow(image)
  plt.axis("off")
  plt.title(title)
  plt.show()

def ScaledAlphaChannelArray(ImagePath):
  A_CHANNEL_RIP = np.asarray(PIL.Image.open(ImagePath))
  A_CHANNEL_SCALED = np.kron(A_CHANNEL_RIP[:, :, 3], np.ones((4, 4)))
  return A_CHANNEL_SCALED

def texture_identify(texture_path):
  if "_default_Trim" in os.path.basename(texture_path):
      texture_type = "Texture_trim"
      return texture_type
  elif "_default_Badge" in os.path.basename(texture_path):
      texture_type = "Texture_Badge"
      return texture_type
  elif "_default_Dirt" in os.path.basename(texture_path):
      texture_type = "Texture_Dirt"
      return texture_type
  elif "_default_Weapon" in os.path.basename(texture_path):
      texture_type = "Texture_Weapon"
      return texture_type
  elif "_default_Eyes" in os.path.basename(texture_path):
      texture_type = "Texture_Eyes"
      return texture_type
  elif "_default_Primary" in os.path.basename(texture_path):
      texture_type = "Texture_Primary"
      return texture_type
  elif "_default_Secondary" in os.path.basename(texture_path):
      texture_type = "Texture_Secondary"
      return texture_type
  elif "_default" in os.path.basename(texture_path):
      texture_type = "Texture_Default"
      return texture_type

def Generate_Badge_Layer(file_path_in, file_path_out):
    Badge_read = np.asarray(PIL.Image.open(file_path_in))
    Badge = np.kron(Badge_read[:, :], np.ones((4, 4)))

    top_corner_y, top_corner_x = np.where(Badge == 255)[0][0], np.where(Badge == 255)[1][1]
    mid_point_y, mid_point_x = int(top_corner_y) + (256 / 2), int(top_corner_x) + (256 / 2)
    start_point_y, start_point_x = mid_point_y + 31, mid_point_x + 31
    end_point_y, end_point_x = mid_point_y - 32, mid_point_x - 32
    start_point_y, start_point_x = int(start_point_y), int(start_point_x)
    end_point_y, end_point_x = int(end_point_y), int(end_point_x)
    Badge[:, :] = 0

    cv2.rectangle(Badge, pt1=(start_point_x, start_point_y), pt2=(end_point_x, end_point_y), color=(255, 255, 255),
                  thickness=-1)

    save_image(Badge, file_path_out)
