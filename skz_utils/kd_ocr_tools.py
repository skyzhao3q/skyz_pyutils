import os_utils
import cv2
import os

def cut_car_no_image(src_img_dir_path, dst_img_dir_path):
    img_path_list = os_utils.get_all_files_pathList(src_img_dir_path, 'jpg')
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        car_no_img = img[182+5:240, 650+8:842]
        dst_img_path = os.path.join(dst_img_dir_path, os_utils.get_fileName_ext(img_path))
        cv2.imwrite(dst_img_path, car_no_img)
        print("[info] {}".format(dst_img_path))

src_img_dir_path = "../../../ksk/kd_ocr/input/form-image_56-59/55"
dst_img_dir_path = "../../../ksk/kd_ocr/input/car_no_img/55"
src_img_dir_path = "../../../ksk/kd_ocr/input/form-image_56-59/57"
dst_img_dir_path = "../../../ksk/kd_ocr/input/car_no_img/57"
os_utils.createDir(dst_img_dir_path)
cut_car_no_image(src_img_dir_path,dst_img_dir_path)
