import os

import spacy

import numpy as np
import pandas as pd
import random as rd

def treat_wine_string(df_wine, columns):
    for column in columns:
        df_wine[column] = df_wine[column].replace(np.nan, '')
        df_wine[column] = df_wine[column].replace('', '_____') # ? dont know if this is the best value to nan

def treat_wine_float(df_wine, columns):
    for column in columns:
        df_wine[column] = df_wine[column].replace(np.nan, 0)
        df_wine[column] = df_wine[column].astype(float)

def load_wine_data():    
    script_dir = os.path.dirname(__file__)
    data_dir = script_dir.replace('src', '') + 'data/'
    wine_file_name = 'winemag-data-130k-v2.csv'
    wine_file = data_dir + wine_file_name 

    df_wine = pd.read_csv(wine_file, index_col=0)

    return df_wine

def get_most_singularity(nlp, values, statement):
    doc1 = nlp(statement)

    similarity = []

    for value in values:
        doc2 = nlp(str(value))
        similarity.append(doc1.similarity(doc2))

    max_similarity = max(similarity)
    max_index = similarity.index(max_similarity)

    return values[max_index]

def get_statement_variety(doc):
    for chunk in doc.noun_chunks:
        if(chunk.root.dep_ == 'dobj' or chunk.root.text == 'wine'):
            return chunk.text # ! problem: assumes that are only one dobj in sentence
    return '_'

def get_statement_price(doc):
    for entity in doc.ents:
        if(entity.label_ == 'MONEY'):
            return entity.text
    return '_'

def df_filter_price_range(df_wine, price_start_range, price_end_range):
    df_wine_filtered = df_wine[
            (df_wine['price'] >= price_start_range) & (df_wine['price'] <= price_end_range)
            ]
    return df_wine_filtered

def apply_variety_filter(df_wine, variety_filter):
    df_wine_filtered = df_wine[
            df_wine['variety'] == variety_filter]
    if(df_wine_filtered.empty == True):
        return df_wine
    return df_wine_filtered

def apply_price_filter(df_wine, price_filter):
    price = extract_numbers_from_string(price_filter)
    if('around' in price_filter):
        # because extract_numbers_from_string can extract a list of values, in the case around, theres only one
        price = price[0]
        price_start_range = price * 0.8 # the range starts in 80% price
        price_end_range = price * 1.2 # and stops in 120% price 
        df_wine_filtered = df_filter_price_range(df_wine, price_start_range, price_end_range)

    elif('between' in price_filter):
        price_start_range = price[0]
        price_end_range = price[1]
        df_wine_filtered = df_filter_price_range(df_wine, price_start_range, price_end_range)

    elif('more' in price_filter):
        price_range = price[0]
        df_wine_filtered = df_wine[
            df_wine['price'] > price_range]

    elif('lass' in price_filter):
        price_range = price[0]
        df_wine_filtered = df_wine[
            df_wine['price'] < price_range]

    else:
        df_wine_filtered = df_wine

    if(df_wine_filtered.empty == True):
        return df_wine

    return df_wine_filtered

def extract_numbers_from_string(string):
    # return [int(s) for s in string.split() if s.isdigit()]
    new_string = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in string)
    return [float(i) for i in new_string.split()]

def chatbot(statement):
    greetings = ['hello', 'hola', 'hi', 'sup', 'wassup', 'whats up', 'oi']
    exit = ['exit', 'bye', 'stop', 'cancel']
    if(statement in greetings):
        return rd.choice(greetings)
    elif(statement in exit):
        return 'bye'

    df_wine = load_wine_data()

    treat_wine_string(df_wine, ['description','designation', 'province', 'region_1', 'region_2',
        'taster_name', 'taster_twitter_handle', 'title', 'variety', 'winery'])
    treat_wine_float(df_wine, ['price'])

    nlp = spacy.load("en_core_web_md")

    statement = 'I want a pinot gris that costs around $25.' # TODO it will be an input

    doc_statement = nlp(statement)

    price_filter = get_statement_price(doc_statement)
    df_filtered = apply_price_filter(df_wine, price_filter)

    varieties = df_wine['variety'].unique()
    variety_statement = get_statement_variety(doc_statement)
    variety_filter = get_most_singularity(nlp, varieties, variety_statement)
    df_filtered = apply_variety_filter(df_filtered, variety_filter)

    title = df_filtered['title'].iloc[0]
    price = df_filtered['price'].iloc[0]
    country = df_filtered['country'].iloc[0]
    title = df_filtered['title'].iloc[0]
    variety = df_filtered['variety'].iloc[0]
    description = df_filtered['description'].iloc[0]

    return f'Recommendation: {title}, {variety} from {country} costs in average ${price}. Its description is: {description}'

def main():
    stop = False
    while not stop:
        statement = input()
        response = chatbot(statement)
        print(response)
        if(response == 'bye'):
            stop = True


if __name__ == '__main__':
    main()