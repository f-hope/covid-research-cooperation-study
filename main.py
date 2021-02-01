# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import re
from operator import itemgetter
import pandas as pd
import pycountry
import json
import os
import re
import pprint
import geograpy
import Levenshtein as lev
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
from nltk.corpus import stopwords
from nltk import ngrams, FreqDist
from collections import Counter

def get_stopwords(stop_words, add_words,n_most_common):
    """
    :param stop_words: starting list of stop words
    :param add_words: list of words from which to add common entries to stop words
    :param n_most_common: number of words from add set to include
    :return: updated list of stop words
    """
    words = ' '.join(add_words)
    words = re.sub(r'\([^)]*\)', '', words)
    tokens = [token.lower() for token in words.split(" ") if token != '']

    counts = Counter(tokens)

    #TODO: make n_most_commmon based on percentage appearance
    [stop_words.append(w[0]) for w in counts.most_common(n_most_common)]
    return stop_words



def stopless_fuzzy(a, b, stopwords):
    """
    :param a: string to be matched
    :param b: string from target set to assess match fit
    :param stopwords: list of words to exclude from match
    :return: match percentage based on set token ratio
    """
    #a = re.sub(r'\([^)]*\)', '', a.lower())
    #b = ' '.join([word.lower() for word in b.split(' ') if word not in stopwords])
    match = process.extractOne(a,b,scorer=fuzz.token_set_ratio)
    return match




def country_match(country, countries, stopwords):
        """
        Match free-text affiliation field to list of institutions
        :param institution: institution free text field to attempt to match
        :param institutions: data frame of Nature institutions
        :param stopwords: list of stopwords to exclude from fuzzy match
        :return:match_institution - best institution match
                match_institution_country - country of best match
                match_institution_match_ratio - ration of the match out of 100

        """

        country_list = countries.clean_country.values.tolist()

        choices_dict = {idx: el for idx, el in enumerate(country_list)}
        match = process.extractOne(country, choices_dict, scorer=fuzz.token_sort_ratio)

        matchloc = match[2]

        #match = [(inst, i[0], i[1], stopless_fuzzy(inst,i[0], stopwords)) for i in institutions_list]
        match_country = countries.iloc[matchloc].location
        match_country_match_ratio = match[1]

        return match_country,match_country_match_ratio


def institution_match(institution, institutions, stopwords):
        """
        Match free-text affiliation field to list of institutions
        :param institution: institution free text field to attempt to match
        :param institutions: data frame of Nature institutions
        :param stopwords: list of stopwords to exclude from fuzzy match
        :return:match_institution - best institution match
                match_institution_country - country of best match
                match_institution_match_ratio - ration of the match out of 100

        """

        institutions_list = institutions.clean_institution.values.tolist()

        inst = ' '.join([word.lower() for word in institution.split(' ') if word.lower() not in stopwords])

        choices_dict = {idx: el for idx, el in enumerate(institutions_list)}
        match = process.extractOne(inst, choices_dict, scorer=fuzz.token_sort_ratio)

        matchloc = match[2]

        #match = [(inst, i[0], i[1], stopless_fuzzy(inst,i[0], stopwords)) for i in institutions_list]
        match_institution = institutions.iloc[matchloc].Institution
        match_institution_country = institutions.iloc[matchloc].Country
        match_institution_match_ratio = match[1]

        return match_institution,match_institution_country,match_institution_match_ratio


def clean_country(country,countries):
    """
    :param country: the country to match to a list of countries
    :param countries: target list of countries
    :return: best match
    """
    country = re.sub(r'\W+', '', country)

    country = country.lower()
    #todo replace with process.extractone
    matches = [(country, c.name, fuzz.ratio(country,c.name.lower())) for c in countries]
    top_match = max(matches,key=itemgetter(2))
    s = pd.Series()

    s['match'] = top_match[2]
    s['match_country'] = top_match[1]
    s['ratio'] = top_match[3]

    return top_match

def studies_with_affiliates(study_dir):
    """
    This method reads a folder directory downloaded from the Kaggle Covid-19 open research dataset challenge
     https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
     and creates a dataframe of study authors and institutions where possible

    :param study_dir: local directory containing JSON files of information parsed from studies
    :return: dataframe of study affiliations
    """

    files = [f for f in os.listdir(study_dir)]
    affiliations = []


    for file in files:
        with open(study_dir + file) as json_file:
            data = json.load(json_file)


        paper_id = data['paper_id']
        authors = data['metadata']['authors']
        author_number = 1
        for author in authors:
            firstNm = author.get('first')
            lastNm = author.get('last')

            institution = author['affiliation'].get('institution')
            laboratory = author['affiliation'].get('laboratory')

            try:
                aff_country = author['affiliation'].get('location').get('country')
                aff_region = author['affiliation'].get('location').get('region')
                aff_settlement = author['affiliation'].get('location').get('settlement')
            except AttributeError:
                aff_settlement = None
                aff_region = None
                aff_country = None

            affiliations.append([paper_id, author_number,firstNm,lastNm,institution,laboratory,aff_country,aff_region,aff_settlement])
            author_number +=1

    affiliationsDF = pd.DataFrame(affiliations, columns=['paper_id','author_number','firstNm','lastNm','institution','laboratory','aff_country','aff_region','settlement'])

    return(affiliationsDF)

def clean_affiliation(s,stopwords):
    """
    prepare institutuion text for matching

    :param s: institutuion string
    :param stopwords: list of stopwords
    :return: prepared string
    """
    if s is not None:
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'[^A-Za-z ]+', '', s)
        s = s.lower()
        s = ' '.join([w for w in s.split(' ') if w not in stopwords])
        s = s.replace('usa','unites states')
        s = s.replace('uk','united kingdom')
        s = s.replace('peoples republic china','china')
        s = s.lstrip().rstrip()
        return s
    else:
        return None


def get_institution(x):
    try:

        if len(x.laboratory) > 2 and len(x.institution) < 2:
            return x.laboratory
        else:
            return x.institution
    except:
        return None


def country_label(x, countries,stopwords):
    if x.match_ratio_country >= 70:
        return x.country_match
    elif x.match_ratio_institution >= 70:# and (x.match_ratio_country < 80 or x.match_ratio_country is None):

        cc = clean_affiliation(clean_affiliation(x.institute_match_country, stopwords=stopwords),countries)

        cm = country_match(cc, countries, stopwords = stopwords)

        return cm[0]

    else:
        return x.clean_country


if __name__ == '__main__':


    os.chdir('/Users/florencehope/OpenData/')
    study_dir = 'data/archive/document_parses/pdf_json/'
    institutions = pd.read_csv('data/export.csv')
    country_data = pd.read_csv('data/coronavirus-data-explorer.csv')


    n_stop = 25


    studies = studies_with_affiliates(study_dir)

    studies['institution'] = studies.apply(lambda x: get_institution(x), axis=1)
    studies_with_affs = studies[((studies['aff_country'].notnull()) & (len(studies['aff_country']) > 1))|((studies['institution'].notnull()) & (len(studies['institution']) > 1))]


    ''' deal with countries'''

    country_data = pd.read_csv('data/coronavirus-data-explorer.csv')
    stopwords= nltk.corpus.stopwords.words('english')

    countries = country_data['location'].drop_duplicates().reset_index()

    stopwords = get_stopwords(nltk.corpus.stopwords.words('english'), countries.location.drop_duplicates().to_list(),
                              n_most_common=0)

    countries['clean_country'] = countries.apply(lambda x: clean_affiliation(x.location, stopwords),axis=1)

    studies_with_affs['clean_country'] = studies_with_affs.apply(lambda x: clean_affiliation(x.aff_country,stopwords),axis=1)

    aff_country_studies = studies_with_affs.pivot_table(index=['clean_country'], aggfunc='size').reset_index().sort_values(0,ascending=False)

    aff_country_studies = aff_country_studies

    aff_country_studies['country_match'], aff_country_studies['match_ratio_country'] = zip(
        *aff_country_studies['clean_country'].apply(country_match, countries=countries, stopwords=stopwords))
    #studies_with_affs = studies_with_affs.merge(aff_country_studies, on='clean_country',how='left')




    ''' deal with institutions'''
    stopwords = get_stopwords(nltk.corpus.stopwords.words('english'), institutions.Institution.to_list(),
                              n_most_common=n_stop)
    studies_with_affs['clean_institution'] = studies_with_affs.apply(lambda x: clean_affiliation(x.institution,stopwords=stopwords),axis=1)

    aff_institutes = studies_with_affs.pivot_table(index=['clean_institution'], aggfunc='size').reset_index().sort_values(0,ascending=False)
    institutions['clean_institution'] = institutions['Institution'].apply(lambda x: clean_affiliation(x,stopwords=stopwords))

    import time

    start = time.time()
    print("hello")

    aff_institutes['institute_match'],  aff_institutes['institute_match_country'], aff_institutes['match_ratio_institution'] = zip(*aff_institutes['clean_institution'].apply(institution_match,institutions=institutions,stopwords=stopwords))
    end = time.time()
    print(end - start)

    matched_studies = studies_with_affs.merge(aff_institutes, on='clean_institution')
    matched_studies = matched_studies.merge(aff_country_studies, on='clean_country', how='left')

    ''' second country match from institution'''

    start = time.time()
    print("hello")

    matched_studies['country_label'] = matched_studies.apply(country_label,countries=countries,stopwords=stopwords, axis=1)
    matched_studies.to_csv('matched_studies_25stop_small2.csv')

    end = time.time()
    print(end - start)
    ''''''
