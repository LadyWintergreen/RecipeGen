import torch
from torch.utils.data import Dataset
import os
import glob
import re

EOS_token = "EOS"

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def process_text_data(path):
    recipes = []
    files = glob.glob(path + "/*.txt")
    for file in files:
        lines = open(file, encoding='utf-8').read().strip().split("END RECIPE")
        for l in lines:
            recipe = text_to_recipe_processing(l)
            if recipe is not None:
                recipes.append(recipe)
    return recipes

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
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

test = process_text_data("Cooking_Dataset/test")
print(test[0])

class RecipeData(Dataset):
    def __init__(self, path):
        self.all_recipes = process_text_data(path)

    def __len__(self):
        return len(self.all_recipes)
    
    def __getitem__(self, ndx):
        item = self.all_recipes[ndx]
        ingredients, steps = tensorsFromPair(item)
        return ingredients, steps

#TODO:
# 1. data processing/preprocessing
# 2. dataset class
# 3. dataloader