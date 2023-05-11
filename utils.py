import torch
from torch.utils.data import Dataset
import os
import glob
import re

EOS_token = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count PAD, SOS, and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def __len__(self):
        return self.n_words

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def build_language(data_path):
    all_ingredients = []
    all_recipes = []
    for path in data_path: #manual path list:
        recipes = process_all_text_data(path)
        for r in recipes:
            all_ingredients.append(r[0])
            all_recipes.append(r[1])
    ingredient_lang = Language("ingredients")
    recipe_lang = Language("recipes")
    for ing in all_ingredients:
        ingredient_lang.addSentence(ing)
    for rec in all_recipes:
        recipe_lang.addSentence(rec)
    return ingredient_lang, recipe_lang
        

def process_all_text_data(path):
    print("Processing text data from {}".format(path))
    recipes = []
    files = glob.glob(path + "/*.txt")
    for file in files:
        lines = open(file, encoding='utf-8').read().strip().split("END RECIPE")
        for l in lines:
            recipe = text_to_recipe_processing(l)
            if recipe is not None:
                recipes.append(recipe)
    return recipes #TODO: build language here as well!!

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def text_to_recipe_processing(line):
    title = re.findall(r'Title: (.*)', line)
    ingredients = re.findall(r'ingredients: (.*)', line)
    steps = re.findall(r'ingredients: .*\n([\s\S]*)', line)
    try:
        title = title[0]
        ingredients = ingredients[0].replace('''\t''', " ")
        steps = steps[0].replace('''\n''', " ")
    except:
        return None
    return (str(title + ingredients), steps)

test = process_all_text_data("Cooking_Dataset/test")
print(test[10])

def collate_fn(batch):
    label_list, text_list = [], [] 
    for (_label, _text) in batch: 
        label_list.append(label_transform(_label)) 
        processed_text = torch.tensor(text_transform(_text)) 
        text_list.append(processed_text) 
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

class RecipeData(Dataset):
    def __init__(self, recipes, ing_lang, instr_lang):
        self.all_recipes = recipes
        self.ing_lang = ing_lang
        self.instr_lang = instr_lang

    def __len__(self):
        return len(self.all_recipes)
    
    def __getitem__(self, ndx):
        item = self.all_recipes[ndx]
        print(ndx)
        print(item)
        ingredients, steps = tensorsFromPair(item, self.ing_lang, self.instr_lang)
        return ingredients, steps



#TODO:
# 1. data processing/preprocessing
# 2. dataset class
# 3. dataloader