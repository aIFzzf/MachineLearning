import numpy as np

csvfilepath = r"../ColorPredict/Dataset/data.csv"
jsonfilepath = r"../ColorPredict/Dataset/JsonColor.json"
import pandas as pd

import json

def num_sum(wine_reviews,name):
    new_num_list = []
    for i in wine_reviews[name]:
        new_i = i.replace("(", ";")
        new_ii = new_i.replace(")", ";").split(";")[1:-2]
        new_num = []

        for num in new_ii:
            new_num.append(float(num))
        new_num_list.append(new_num)


    return new_num_list

def get_color(name):
    new_name = name.replace(",", ";")[1:-3].split(";")[0:-1]

    new_num = []
    for num in new_name:
        new_num.append(float(num))

    return new_num


def get_num(wine_reviews,name):
    num_list = []
    for i in wine_reviews[name]:
        num_list.append(i)

    return num_list


def clamp(num,a,b):
    return max(min(num,max(a,b)),min(a,b))

def read_data():
    wine_reviews = pd.read_csv(csvfilepath)

    # read data from csv
    color1_list = num_sum(wine_reviews, "Color1")
    color1_array = np.array(color1_list)
    color2_list = num_sum(wine_reviews, "Color2")
    color2_array = np.array(color2_list)
    blendcolor_list = num_sum(wine_reviews, "BlendColor")
    blendcolor_array = np.array(blendcolor_list)

    lerp_list = get_num(wine_reviews,"Lerp")
    lerp_array = np.array(lerp_list)


    return color1_list,color1_array,color2_list,color2_array,blendcolor_list,blendcolor_array,lerp_list,lerp_array

def read_json():
    with open(jsonfilepath, 'r') as file:
        data = json.load(file)
        color1_list =[]
        color2_list =[]
        blendcolor_list =[]
        lerp_list = []
        for item in data :
            color1_list.append(get_color(item.get("blendColor1")))
            color2_list.append(get_color(item.get("blendColor2")))
            blendcolor_list.append(get_color(item.get("outColor")))
            lerp_list.append(float(item.get("blend")))

        color1_array = np.array(color1_list)
        color2_array = np.array(color2_list)
        blendcolor_array = np.array(blendcolor_list)
        lerp_array = np.array(lerp_list)
    return color1_list,color1_array,color2_list,color2_array,blendcolor_list,blendcolor_array,lerp_list,lerp_array


def toShader(feature_list,reg_coef):


    feature_str_list = []
    for feature in feature_list:
        temp = ""
        templist1 = feature.split("^")
        templist2 = feature.split(" ")
        if(len(templist1) == 2):
            temp =  templist1[0] + "*" + templist1[0]
        elif(len(templist2) == 2):
            temp = templist2[0] + "*" + templist2[1]
        else:
            temp = feature

        feature_str_list.append(temp)


    R_channel = "float r = "
    G_channel = "float g = "
    B_channel = "float b = "
    # A_channel = "float a = "
    # RGBA_list = [R_channel,G_channel,B_channel,A_channel]
    RGBA_list = [R_channel,G_channel,B_channel]
    shader = "float x0 = color1.r;\n float x1 = color1.g; \n float x2 = color1.b;\n " \
             "float x3 = color2.r;\n float x4 = color2.g; \n float x5 = color2.b;\n " \
             "float x6 = blend;\n"
    for i  in range(0,len(RGBA_list),1):
        channel = ""
        for j in range(0,len(reg_coef[i]),1):
            coef = reg_coef[i][j]
            coef_num = as_num(coef)
            if(j == 0):
                channel += "("+(str(coef_num) + "*" + feature_str_list[j])+")"
            else:
                channel += "+" + "("+(str(coef_num) + "*" + feature_str_list[j])+")"

        shader += RGBA_list[i] + channel + ";"
        shader += "\n"

    shader += "float3 color = float3(r,g,b);\n return color;"
    print(shader)

def as_num(x):
    y = '{:.2f}'.format(x)  # .2f 保留2位小数
    return y