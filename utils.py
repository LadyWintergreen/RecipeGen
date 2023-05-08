import torch
from torch.utils.data import Dataset
import os
import glob
import re

EOS_token = "EOS"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def build_language(data_path):
    all_ingredients = []
    all_recipes = []
    for path in data_path: #manual path list:
        recipes = process_text_data(path)
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
        

def process_text_data(path):
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

test = process_text_data("Cooking_Dataset/test")
print(test[0])

def collate_fn(data): # may be able to just pack instead
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths
    #Goal is to create minibatch tensors from single batch data
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths


class RecipeData(Dataset):
    def __init__(self, path, ing_lang, instr_lang):
        self.all_recipes = process_text_data(path)
        self.ing_lang = ing_lang
        self.instr_lang = instr_lang

    def __len__(self):
        return len(self.all_recipes)
    
    def __getitem__(self, ndx):
        item = self.all_recipes[ndx]
        ingredients, steps = tensorsFromPair(item, self.ing_lang, self.instr_lang)
        return ingredients, steps



#TODO:
# 1. data processing/preprocessing
# 2. dataset class
# 3. dataloader