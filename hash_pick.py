import os
from PIL import Image
import torch
import shutil

resize_height = 9
resize_width = 9
judge_thresh = 10
img_root = '/home/wty/data/YSData/train'
img_dhash_pick_root = '/home/wty/data/YSData/train_dhash'
img_dhash_pick_file_dir = '/home/wty/data/YSData/dhash_pick.txt'
img_dhash_dict = {}


def main():
    
    # create hash dict and calculate hash for each image
    for (_, folders, files) in os.walk(img_root):
        folders.sort()
        for (img_cls_num, folder) in enumerate(folders):
            img_cls = folder
            img_dhash_dict[img_cls] = {}
            img_cls_dir = os.path.join(img_root, img_cls)
            img_name_list = os.listdir(img_cls_dir)
            img_name_list.sort()
            img_dhash_dict[img_cls]['img_list'] = img_name_list
            img_dhash_dict[img_cls]['hash_list'] = []
            for (img_num, img_name) in enumerate(img_name_list):
                if not img_num % 500:
                    print('creating hash dict: img_class: {} / {} : {}  img: {} / {}'.format(
                        img_cls_num+1, len(folders), img_cls, img_num+1, len(img_name_list)))
                img_dir = os.path.join(img_cls_dir, img_name)
                img = Image.open(img_dir)
                # resize image to small size
                img_resize = img.resize((resize_width, resize_height), Image.ANTIALIAS)
                # convert to grey
                img_grey = img_resize.convert("L")
                img_hash_string = calculate_hash(img_grey)
                img_dhash_dict[img_cls]['hash_list'].append(img_hash_string)
    
    with open(img_dhash_pick_file_dir, 'a') as fout:
        for (img_cls_id, img_cls) in enumerate(img_dhash_dict.keys()):
            img_num = len(img_dhash_dict[img_cls]['img_list'])
            img_hash_matrix = torch.zeros(img_num, img_num)
            print('calculating hash matrix for class {} / {}: {}'.format(img_cls_id+1, len(img_dhash_dict.keys()), img_cls))
            for i in range(img_num):
                hash_string_1 = img_dhash_dict[img_cls]['hash_list'][i]
                for j in range(i+1, img_num):
                    hash_string_2 = img_dhash_dict[img_cls]['hash_list'][j]
                    dhash = hamming_distance(hash_string_1, hash_string_2)
                    img_hash_matrix[i][j] = dhash
                    if dhash < judge_thresh:
                        file_name_1 = img_dhash_dict[img_cls]['img_list'][i]
                        file_name_2 = img_dhash_dict[img_cls]['img_list'][j]
                        fout.write('{} {}\n'.format(file_name_1, file_name_2))
                        file_cls_dir = os.path.join(img_dhash_pick_root, img_cls)
                        if not os.path.exists(file_cls_dir):
                            os.makedirs(file_cls_dir)
                        file_src_1 = os.path.join(img_root, img_cls, file_name_1)
                        file_dst_1 = os.path.join(file_cls_dir, file_name_1)
                        file_src_2 = os.path.join(img_root, img_cls, file_name_2)
                        file_dst_2 = os.path.join(file_cls_dir, file_name_2)
                        if not os.path.exists(file_dst_1):
                            shutil.copyfile(file_src_1, file_dst_1)
                        shutil.copyfile(file_src_2, file_dst_2)
                        
                



    # img_1 = '/home/wty/data/YSData/img1.jpg'
    # img_2 = '/home/wty/data/YSData/img2.jpg'
    # img_1 = '/home/wty/data/YSData/train/apron/apron_00001.jpg'
    # img_2 = '/home/wty/data/YSData/train/apron/apron_00029.jpg'
    # resize_height = 9
    # resize_width = 9
    # img_1 = Image.open(img_1)
    # img_2 = Image.open(img_2)
    # img_1 = img_1.resize((resize_width, resize_height), Image.ANTIALIAS)
    # img_2 = img_2.resize((resize_width, resize_height), Image.ANTIALIAS)
    # img_1 = img_1.convert("L")
    # img_2 = img_2.convert("L")
    #
    # hash_1 = calculate_hash(img_1)
    # hash_2 = calculate_hash(img_2)
    # dhash = hamming_distance(hash_1, hash_2)
    # print(dhash)
    

def calculate_hash(image):
    """
    计算图片的dHash值
    :param image: PIL.Image
    :return: dHash值,string类型
    """
    difference = hash_difference(image)
    # 转化为16进制(每个差值为一个bit,每8bit转为一个16进制)
    decimal_value = 0
    hash_string = ""
    for index, value in enumerate(difference):
        if value:  # value为0, 不用计算, 程序优化
            decimal_value += value * (2 ** (index % 8))
        if index % 8 == 7:  # 每8位的结束
            hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # 不足2位以0填充。0xf=>0x0f
            decimal_value = 0
    return hash_string


def hamming_distance(first, second):
    """
    计算两张图片的汉明距离(基于dHash算法)
    :param first: Image或者dHash值(str)
    :param second: Image或者dHash值(str)
    :return: hamming distance. 值越大,说明两张图片差别越大,反之,则说明越相似
    """
    # A. dHash值计算汉明距离
    if isinstance(first, str):
        return hamming_distance_with_hash(first, second)

    # B. image计算汉明距离
    hamming_distance = 0
    image1_difference = hash_difference(first)
    image2_difference = hash_difference(second)
    for index, img1_pix in enumerate(image1_difference):
        img2_pix = image2_difference[index]
        if img1_pix != img2_pix:
            hamming_distance += 1
    return hamming_distance


def hash_difference(image):
    """
    *Private method*
    计算image的像素差值
    :param image: PIL.Image
    :return: 差值数组。0、1组成
    """
    resize_width = 9
    resize_height = 8
    # 1. resize to (9,8)
    smaller_image = image.resize((resize_width, resize_height))
    # 2. 灰度化 Grayscale
    grayscale_image = smaller_image.convert("L")
    # 3. 比较相邻像素
    pixels = list(grayscale_image.getdata())
    difference = []
    for row in range(resize_height):
        row_start_index = row * resize_width
        for col in range(resize_width - 1):
            left_pixel_index = row_start_index + col
            difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
    return difference


def hamming_distance_with_hash(dhash1, dhash2):
    """
    *Private method*
    根据dHash值计算hamming distance
    :param dhash1: str
    :param dhash2: str
    :return: 汉明距离(int)
    """
    difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
    return bin(difference).count("1")


if __name__ == '__main__':
    main()
