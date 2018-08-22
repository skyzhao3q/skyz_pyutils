from . import os_utils
from . import ds_utils
#import os_utils
#import ds_utils
import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def read(img_file_path, mode=('rgb', 'gray', 'bin'), threshold_val=None):
    ''' read image
    '''
    if mode not in ('rgb', 'gray', 'bin'):
        assert False, "<im_utils> Invalid Mode: {}".format(mode)
    if mode == 'rgb':
        img = cv2.imread(img_file_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if mode == 'gray':
        img = cv2.imread(img_file_path, 0)
    if mode == 'bin':
        if threshold_val is None:
            assert False, "<im_utils> NO threshold value"
        img = cv2.imread(img_file_path, 0)
        _,img = cv2.threshold(img,threshold_val,255,cv2.THRESH_BINARY)
    
    return img

def write(img_file_path, img):
    _,_,c = img.shape
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_file_path, img)

def rotate(img, angle):
    img = imutils.rotate(img, angle=angle)

    return img

def rotate_bound(img, angle):
    '''rotate without cut off image
       fill with black pixel
    '''
    img = imutils.rotate_bound(img, angle=angle)

    return img

def rotate_bound_white(img, angle):
    '''rotate without cut off image
       fill with white pixel
    '''
    # convert white 2 black
    img = ~img
    # fill with black
    img = imutils.rotate_bound(img, angle=angle)
    # recover 2 white
    img = ~img

    return img

def resize_with_aspect(img, width=None, height=None):
    img = imutils.resize(img, width=width, height=height)

    return img

def bgr2rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def show_img(img, title=None):
    if title is not None:
        plt.figure(title)
    plt.imshow(img)
    plt.show()

def show_gray_img(img, title=None, vmin=0, vmax=255):
    if title is not None:
        plt.figure(title)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()

def dir_image(image_dir_path, result_dir_path, batch=10, ext='png'):
    one_image = None
    image_path_list = os_utils.get_all_files_pathList(image_dir_path, ext)
    image_path_set = [image_path_list[i:i+batch] for i in range(0, len(image_path_list), batch)]
    input_shape = (64, 64, 3)
    for sub_image_list in image_path_set:
        images_row = np.zeros((input_shape[0], input_shape[1]*batch, input_shape[2]), dtype=np.uint8)
        sub_image_data = ds_utils.image_dataset_from(sub_image_list, input_shape)
        sub_image_data = np.concatenate((sub_image_data), axis=1)
        images_row[0:sub_image_data.shape[0], 0:sub_image_data.shape[1], 0:sub_image_data.shape[2]] = sub_image_data
        if one_image is not None:
            one_image = np.concatenate((one_image, images_row), axis=0)
        else:
            one_image = images_row
    one_image = cv2.cvtColor(one_image, cv2.COLOR_RGB2BGR)
    image_file_path = result_dir_path + "/" + "one_image-" + os_utils.get_dir_name(image_dir_path) + "_" + str(len(image_path_list)) + ".png"
    cv2.imwrite(image_file_path, one_image)

    return (image_file_path)

def multi_dir_image(image_dir_path, result_dir_path, batch=10, ext='png'):
    sub_image_dir_path_list = os_utils.get_dir_path_list(image_dir_path)
    for sub_image_dir_path in sub_image_dir_path_list:
        yield dir_image(sub_image_dir_path, result_dir_path, batch=batch, ext=ext)

def resize_with_padding(src, dst_size, padding_color=[255,255,255]):
    src_w = src.shape[1]
    src_h = src.shape[0]
    dst_w, dst_h = dst_size
    if src_w > dst_w or src_h > dst_h:
        src = resize_with_aspect(src, width=dst_w,  height=dst_h)
    src_w = src.shape[1]
    src_h = src.shape[0]
    delta_w = dst_w  - src_w
    delta_h = dst_h - src_h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    dst = cv2.copyMakeBorder(src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

    return dst