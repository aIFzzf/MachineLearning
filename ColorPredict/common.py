import numpy as np

csvfilepath = r"../ColorPredict/Dataset/data.csv"
import pandas as pd



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

