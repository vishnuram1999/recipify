from flask import Flask
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import requests
import json


model = load_model('food_model.h5')

food_category = np.load('food_category.npy')


app = Flask(__name__)


def get_food_nutrients(food):
    url = "https://nutritionix-api.p.rapidapi.com/v1_1/search/{}".format(food)

    querystring = {"fields": "item_name,nf_calories,nf_total_fat"}

    headers = {
        'x-rapidapi-host': "nutritionix-api.p.rapidapi.com",
        'x-rapidapi-key': "e4573ac34fmshc71719554be1369p1d0fcajsnc45c98c12e74"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    # print(response.text)

    output = response.json()

    calories = []
    fat = []

    hits = output["hits"]
    calories = hits[0]["fields"]["nf_calories"]
    fat = hits[0]["fields"]["nf_total_fat"]

    nutrients = {"calories": calories,
                 "fat": fat}

    # nutrients = json.dumps(res)

    return nutrients


def get_food_details(food):
    url = "https://recipe-puppy.p.rapidapi.com/"

    headers = {
        'x-rapidapi-host': "recipe-puppy.p.rapidapi.com",
        'x-rapidapi-key': "e4573ac34fmshc71719554be1369p1d0fcajsnc45c98c12e74"
    }

    querystring = {"q": str(food)}

    response = requests.request("GET", url, headers=headers, params=querystring)

    output = response.json()

    # print(output['results'])
    ingredients = []
    names = []
    recepies = []

    for food in output['results']:
        content = food['ingredients'].split(',')
        names.append(food['title'])
        recepies.append(food['href'])
        ingredients = ingredients + content

    # res = dict(zip(names, recepies))

    # print(str(res))

    food_recipe = {"names": names,
                   "recepies": recepies,
                   "ingredients": ingredients}

    return food_recipe

def get_json(food):

    food_recipe = get_food_details(food)
    print(food_recipe)
    nutrients = get_food_nutrients(food)
    #print(nutrients)

    total_dict = {"prediction" : food, **food_recipe, **nutrients}

    #print(total_dict)

    output = json.dumps(total_dict)

    return output



def preprocessing(path):

    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = image/255
    image = np.reshape(image, [-1, 128, 128, 3])

    return image

def predict(path):

    image = preprocessing(path)
    food = food_category[model.predict(image).argmax() - 1]
    food = food.replace("_", " ")

    return food


@app.route('/')
def index():
    return "Hello World"

@app.route('/home')
def send_data():

    image_path = 'french-fries.jpg'

    food = predict(image_path)

    output = get_json(food)

    return output