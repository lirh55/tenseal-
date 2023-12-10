import tenseal as ts
import numpy as np
from PIL import Image
import math
import time

def gencontext():
    context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193)
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

def encrypt(context, np_vector):
    return ts.bfv_vector(context, np_vector)

def decrypt(enc_vector):
    return np.array(enc_vector.decrypt())

def load_image(image_path):
    img = Image.open(image_path).convert('L')
    img_vector = np.asarray(img).flatten()
    return img_vector

def save_encrypted_data(enc_vector, file_path):
    with open(file_path, 'wb') as file:
        file.write(enc_vector.serialize())

def load_encrypted_vector(file_path, context, batch=None):
    with open(file_path, 'rb') as file:
        serialized_data = file.read()
        vector = ts.bfv_vector_from(context, serialized_data)
    return vector

def MMSE(ori, tar):
    temp = ori - tar
    result = temp * temp
    return result

def search(file_path_folder, img_path, context):
    img_data = load_image(img_path)
    encrypted_img = encrypt(context, img_data)
    time_list = []
    for i in range(0, 15):
        # print("正在与第{}张图片进行匹配".format(i+1))
        file_path = file_path_folder + str(i) + ".dat"
        encrypted_data = load_encrypted_vector(file_path, context)
        t = time.time()
        en_numerator = encrypted_img.dot(encrypted_data)
        en_numerator = en_numerator.dot(en_numerator)
        en_denominator_a = encrypted_data.dot(encrypted_data)
        en_denominator_b = encrypted_img.dot(encrypted_img)
        numerator = decrypt(en_numerator)
        denominator = decrypt(en_denominator_a.dot(en_denominator_b))
        result_square = numerator[0] / denominator[0]
        if (result_square > 0):
            result = math.sqrt(result_square)
            if abs(result - 1) < 0.0001:
                number = i
        timeend = time.time() - t
        time_list.append(timeend)
        # print("对第{}张图片使用余弦距离进行匹配用时为{}s\n".format(i + 1, timeend))
    print("最佳匹配图片为第{}张，检索总计用时{}s，平均单张图片使用余弦距离进行匹配用时{}".format(number, np.sum(time_list),
                                                                                             np.mean(time_list)))

if __name__ == "__main__":
    context = gencontext()
    image_folder = r".\64test"
    file_path_folder = r".\encrypted_data\encrypted_data"
    target_image_path = r".\64test\test_8.JPEG"

    time_list_en = []
    time_list_de = []
    for i in range(0, 15):
        image_path = image_folder + "\\test_" + str(i) + ".JPEG"
        img_data = load_image(image_path)
        t = time.time()
        enc_a = encrypt(context, img_data)
        timeend = time.time() - t
        time_list_en.append(timeend)
        print("加密第{}张图片用时{}s".format(i + 1, timeend))
        t = time.time()
        de_a = decrypt(enc_a)
        timeend = time.time() - t
        time_list_de.append(timeend)
        print("解密第{}张图片用时{}s\n".format(i + 1, timeend))
        file_path = file_path_folder + str(i) + ".dat"
        save_encrypted_data(enc_a, file_path)
    print("平均单站图片加密用时{}s，平均单张图片解密用时{}s".format(np.mean(time_list_en), np.mean(time_list_de)))
    search(file_path_folder, target_image_path, context)
