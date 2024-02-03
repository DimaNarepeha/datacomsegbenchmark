import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from loadModel import sam_vit_h_4b8939_predictor_with_cuda

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def fillColor(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_overlaid_figure(manual_mask, mask, title):
    plt.figure(figsize=(10,10))
    plt.imshow(manual_mask)
    show_mask(mask, plt.gca())
    plt.title(title, fontsize=18)
    plt.savefig(title)
    plt.show()
#    fig.savefig('my_figure.png')

def generate_mask(predictor, image, input_point, input_label):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return max(zip(scores, masks))
#    for i, (mask, score) in enumerate(zip(masks, scores)):
#        plt.figure(figsize=(10,10))
#        plt.imshow(image)
#        show_mask(mask, plt.gca())
#        show_points(input_point, input_label, plt.gca())
#        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#        plt.show()

def getCompressionRatio(image_og, compressed_img):
    # Calculate compression rate
    # Calculate sizes of the numpy arrays
    original_size = image_og.nbytes
    compressed_size = compressed_img.nbytes
    print(f"the original image size is: {original_size}")
    print(f"the compressed image size is: {compressed_size}")

    # Calculate compression rate
    compression_rate = ((original_size - compressed_size) / original_size) * 100
    print(f"Compression Rate: {compression_rate:.2f}%")

def convert_png_to_jpg(png_file_path, jpg_file_path, quality):
    # Read the image from the PNG file
    image = cv2.imread(png_file_path, cv2.IMREAD_UNCHANGED)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Unable to load image.")
        return
    # Convert the image to JPG and save it
    cv2.imwrite(jpg_file_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def encoded_from_JPEG(image, quality):
    is_success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not is_success:
        raise Exception("JPEG Encoding Error Occured")
    return buffer

def decoded_from_buffer(buffer):
    image_array = np.frombuffer(buffer, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    return image

def calculate_compression_ratio(image, buffer):
    uncompressed_size = image.shape[0] * image.shape[1] * (12 / 8)
    print(f'the uncompressed file size is {uncompressed_size} bytes')
    compressed_size = buffer.nbytes
    print(f'after compression, the file size is {compressed_size} bytes')
    compression_ratio = uncompressed_size / compressed_size
    return compression_ratio

def startPrediction(predictor, image_path, mask_path, coords, coords_labels, quality, title):
    image = cv2.imread(image_path)
    buffer = encoded_from_JPEG(image, quality)
    print(f'the compression ratio is {calculate_compression_ratio(image, buffer)}')
    compressed_image = decoded_from_buffer(buffer)
    max_score, max_mask = generate_mask(predictor, compressed_image , coords, coords_labels)
    manual_mask_image = cv2.imread(mask_path)
    save_overlaid_figure(manual_mask_image, max_mask, title=f"{title}_{max_score}.png")

if __name__ == "__main__":

    LEFT_MASK_PATH = "./LungImages/leftMask/MCUCXR_0003_0.png"
    RIGHT_MASK_PATH= "./LungImages/rightMask/MCUCXR_0003_0.png"
    DATA_PATH = "./LungImages/dataset/0/MCUCXR_0003_0.png"
    IMAGE_SIZE = (4892, 4020)
    JPEG_QUALITY = [20, 40, 60, 80]

    og_image = cv2.imread(DATA_PATH)

    left_input_point = np.array([[1352, 1308], [983, 2160]])
    left_input_label = np.array([1, 1])
    
#    plt.figure(figsize=(10,10))
#    plt.imshow(og_image)
#    show_points(left_input_point, left_input_label, plt.gca())
#    plt.title("prompts of the left lung", fontsize=18)
#    plt.savefig("prompts of the left lung")
#    plt.show()

#    right_input_point = np.array([[2851, 1226], [3239, 2217]])
#    right_input_label = np.array([1, 1])
#    plt.figure(figsize=(10,10))
#    plt.imshow(og_image)
#    show_points(right_input_point, right_input_label, plt.gca())
#    plt.title("prompts of the right lung", fontsize=18)
#    plt.savefig("prompts of the right lung")
#    plt.show()
#
#
#initialize the model
    predictor = sam_vit_h_4b8939_predictor_with_cuda()

    new_left_input_point = np.array([[1352, 1308], [983, 2160], [1739, 1842], [1759, 1601]])
    new_left_input_label = np.array([1, 1, 0, 0])
    plt.figure(figsize=(10,10))
    plt.imshow(og_image)
    show_points(new_left_input_point, new_left_input_label, plt.gca())
    plt.title("prompts of the left lung", fontsize=18)
    plt.savefig("prompts of the left lung")
    plt.show()

    max_score, max_mask = generate_mask(predictor, og_image , new_left_input_point, new_left_input_label)
    manual_mask_image = cv2.imread(LEFT_MASK_PATH)
    save_overlaid_figure(manual_mask_image, max_mask, title=f"new_OG_LEFT_{max_score}.png")
##
#    for quality in JPEG_QUALITY:
#        startPrediction(predictor, DATA_PATH, LEFT_MASK_PATH, left_input_point, left_input_label, quality, title=f"left_{quality}")
#
    
    #Right Mask
#   max_score, max_mask = generate_mask(predictor, og_image , right_input_point, right_input_label)
#   manual_mask_image = cv2.imread(RIGHT_MASK_PATH)
#   save_overlaid_figure(manual_mask_image, max_mask, title=f"OG_RIGHT_{max_score}.png")

#    for quality in JPEG_QUALITY:
#        startPrediction(predictor, DATA_PATH, RIGHT_MASK_PATH, right_input_point, right_input_label, quality, title=f"right_{quality}")



