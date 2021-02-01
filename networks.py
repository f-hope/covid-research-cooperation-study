import networkx as nx
from networkx.algorithms import community


import plotly.graph_objects as go
import pandas as pd


import networkx as nx

G = nx.random_geometric_graph(200, 0.125)



if __name__ == '__main__':

    os.chdir('/Users/florencehope/OpenData/')
    aff_matched = pd.read_csv('matched_studies_100stop.csv')
    study_metadata = pd.read_csv('data/archive/metadata.csv')

    affs = aff_matched[['paper_id','author_number','clean_institution','institute_match','institute_match_country','match_ratio']]

    affs['institute'] = affs.apply(institution_label,axis=1)
    affs['country'] = affs.apply(country_label, axis = 1)

    grpAffsInstitute = group_by_key(affs,'institute')

    grpAffsCountry = group_by_key(affs,'country')