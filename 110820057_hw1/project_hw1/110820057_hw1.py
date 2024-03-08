import cv2
import numpy as np

def write_img(file_name, img):
    file_path = 'results/{}'.format(file_name)
    cv2.imwrite(file_path, img)

def q1_1(img_name, img):
    rows, cols, _ = img.shape
    for i in range(rows):
        for j in range(cols):
            gray = (0.3 * img[i, j][0]) + (0.59 * img[i, j][1]) + (0.11 * img[i, j][2])
            img[i, j] = [gray, gray, gray]
    write_img('{}_q1-1.jpg'.format(img_name), img)
    return img

def q1_2(img_name, img, threshold = 127):
    rows, cols, _ = img.shape
    for i in range(rows):
        for j in range(cols):
            img[i, j] = [255, 255, 255] if img[i, j][0] >= threshold else [0, 0, 0]
    write_img('{}_q1-2.jpg'.format(img_name), img)

def q1_3(img_name, img):
    pass

def q2_1_upscale(img_name, img):
    rows, cols, _ = img.shape
    result = np.zeros((rows * 2, cols * 2, 3), np.uint8)
    y = 0
    for i in range(rows):
        x = 0
        for j in range(cols):
            result[y, x] = result[y, x + 1] = img[i, j]
            result[y + 1, x] = result[y + 1, x + 1] = img[i, j]
            x += 2
        y += 2
    write_img('{}_q2-1-double.jpg'.format(img_name), result)


def q2_1_downscale(img_name, img):
    rows, cols, _ = img.shape
    result = np.zeros((int(rows / 2), int(cols / 2), 3), np.uint8)
    rows, cols, _ = result.shape
    y = 0
    for i in range(rows):
        x = 0
        for j in range(cols):
            result[i, j] = img[y, x]
            x += 2
        y += 2
    write_img('{}_q2-1-half.jpg'.format(img_name), result)

def q2_2_upscale(img_name, img):
    rows, cols, _ = img.shape
    result = np.zeros((rows * 2 + 1, cols * 2 + 1, 3), np.uint8)
    
    # top left
    y = 0
    for i in range(rows):
        x = 0
        for j in range(cols):
            result[y, x] = img[i, j]
            x += 2
        y += 2

    # top right
    y = 0
    for i in range(rows):
        x = 1
        for j in range(cols):
            result[y, x][0] = (int(result[y, x - 1][0]) + int(result[y, x + 1][0])) / 2
            result[y, x][1] = (int(result[y, x - 1][1]) + int(result[y, x + 1][1])) / 2
            result[y, x][2] = (int(result[y, x - 1][2]) + int(result[y, x + 1][2])) / 2
            x += 2
        y += 2

    # bottom left
    y = 1
    for i in range(rows):
        x = 0
        for j in range(cols):
            result[y, x][0] = (int(result[y - 1, x][0]) + int(result[y + 1, x][0])) / 2
            result[y, x][1] = (int(result[y - 1, x][1]) + int(result[y + 1, x][1])) / 2
            result[y, x][2] = (int(result[y - 1, x][2]) + int(result[y + 1, x][2])) / 2
            x += 2
        y += 2

    # bottom right
    y = 1
    for i in range(rows):
        x = 1
        for j in range(cols):
            result[y, x][0] = (int(result[y - 1, x - 1][0]) + int(result[y - 1, x + 1][0]) + int(result[y + 1, x - 1][0]) + int(result[y + 1, x + 1][0])) / 4
            result[y, x][1] = (int(result[y - 1, x - 1][1]) + int(result[y - 1, x + 1][1]) + int(result[y + 1, x - 1][1]) + int(result[y + 1, x + 1][1])) / 4
            result[y, x][2] = (int(result[y - 1, x - 1][2]) + int(result[y - 1, x + 1][2]) + int(result[y + 1, x - 1][2]) + int(result[y + 1, x + 1][2])) / 4
            x += 2
        y += 2

    result = result[0:rows * 2, 0:cols * 2]
    write_img('{}_q2-2-double.jpg'.format(img_name), result)

def q2_2_downscale(img_name, img):
    rows, cols, _ = img.shape
    result = np.zeros((int(rows / 2), int(cols / 2), 3), np.uint8)
    rows, cols, _ = result.shape
    y = 0
    for i in range(rows):
        x = 0
        for j in range(cols):
            for p in range(3):
                result[i, j][p] = (int(img[y, x][p]) + int(img[y, x + 1][p]) + int(img[y + 1, x][p]) + int(img[y + 1, x + 1][p])) / 4
            x += 2
        y += 2
    write_img('{}_q2-2-half.jpg'.format(img_name), result)

if __name__ == '__main__':
    img1 = cv2.imread('images/img1.png')
    img2 = cv2.imread('images/img2.png')
    img3 = cv2.imread('images/img3.png')
    
    img1_q1_1 = q1_1('img1', img1.copy())
    img2_q1_1 = q1_1('img2', img2.copy())
    img3_q1_1 = q1_1('img3', img3.copy())

    q1_2('img1', img1_q1_1)
    q1_2('img2', img2_q1_1)
    q1_2('img3', img3_q1_1)

    q2_1_upscale('img1', img1)
    q2_1_upscale('img2', img2)
    q2_1_upscale('img3', img3)

    q2_1_downscale('img1', img1)
    q2_1_downscale('img2', img2)
    q2_1_downscale('img3', img3)

    # it would take a while
    q2_2_upscale('img1', img1)
    q2_2_upscale('img2', img2)
    q2_2_upscale('img3', img3)

    q2_2_downscale('img1', img1)
    q2_2_downscale('img2', img2)
    q2_2_downscale('img3', img3)

    print('finish')