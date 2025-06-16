import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取目录中的所有图像
def read_directory(directory_name):
    img_list = []
    for root, dirs, files in os.walk(directory_name):
        for filename in files:
            path = os.path.join(root, filename)
            category = os.path.basename(root)  # 获取子目录名称作为类别
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            img_list.append((img, category, filename))
    return img_list

# 图像预处理
def preprocess_image(image):
    # 直方图均衡化
    image = cv2.equalizeHist(image)
    # 高斯模糊
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # 自适应二值化
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary_image

# 调整图像尺寸以匹配模板
def size_matching(img, template):
    h, w = template.shape
    img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img_resized

# 模板匹配
def match_scoring(image, template, method=cv2.TM_CCOEFF_NORMED):
    result = cv2.matchTemplate(image, template, method)
    score = result[0][0]
    return score

# 识别字符
def recognize_characters(test_images, template_images, method=cv2.TM_CCOEFF_NORMED):
    recognized_characters = []
    for test_img, test_category, test_filename in test_images:
        # 根据文件名中的序号确定字符类型
        char_index = int(test_filename.split('word')[1].split('.')[0])
        if char_index == 1:
            # 第一个字符一定是汉字
            valid_templates = [(img, cat, filename) for img, cat, filename in template_images if cat in '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵青藏川宁琼使领']
        elif char_index == 2:
            # 第二个字符一定是大写字母
            valid_templates = [(img, cat, filename) for img, cat, filename in template_images if cat in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        else:
            # 其他字符可以是数字或大写字母
            valid_templates = [(img, cat, filename) for img, cat, filename in template_images if cat in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        
        scores = []
        for template_img, template_category, template_filename in valid_templates:
            test_img_resized = size_matching(test_img, template_img)
            score = match_scoring(test_img_resized, template_img, method)
            scores.append((template_category, template_filename, score))
        
        scores.sort(key=lambda x: x[2], reverse=True)  # 按得分降序排序
        recognized_characters.append((test_category, test_filename, scores[:5]))  # 取前五名
    return recognized_characters

# 主函数
def main():
    # 读取模板图像
    #template_directory = 'D:/shuzituxiang/final/template_matching/learn_data'      #读取学习匹配的学习集来当模板做对照试验
    template_directory = 'D:/shuzituxiang/final/template_matching/template_data'  #读取模板集
    template_images = read_directory(template_directory)
    
    # 读取测试图像
    test_directory = 'D:/shuzituxiang/final/template_matching/test_data'
    test_images = read_directory(test_directory)
    
    # 预处理测试图像和模板图像
    preprocessed_test_images = [(preprocess_image(img), category, filename) for img, category, filename in test_images]
    preprocessed_template_images = [(preprocess_image(img), category, filename) for img, category, filename in template_images]
    
    # 识别字符
    recognized_characters = recognize_characters(preprocessed_test_images, preprocessed_template_images, method=cv2.TM_CCOEFF_NORMED)
    
    # 输出识别结果
    for i, (test_category, test_filename, top_scores) in enumerate(recognized_characters):
        print(f"Test Image: {test_category}/{test_filename}")
        for template_category, template_filename, score in top_scores:
            print(f"  Recognized as: {template_category}/{template_filename} (Score: {score})")
        print()

if __name__ == "__main__":
    main()