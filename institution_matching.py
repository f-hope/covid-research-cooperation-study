from collections import Counter
import pandas as pd
import rapidfuzz as rfuzz
import os
import matplotlib.pyplot as plt

def group_by_key(affs, key):

    grpAffs = affs.groupby('paper_id').apply(lambda x: num_affiliaties(x, countfield=key)).reset_index()
    grpAffs['num_' + key] = grpAffs.apply(lambda x: len(x[0]), axis=1)
    grpAffs['collaborative_ind_' + key] = grpAffs['num_' + key] > 1

    return grpAffs


def num_affiliaties(study, countfield):

    aff_array = study[countfield].drop_duplicates().values.tolist()

    return aff_array


def institution_label(df):

  if df['match_ratio_institution'] <= 80:
    return df.clean_institution
  else:
      return df.institute_match


def institution_label(df):

  if df['match_ratio_institution'] <= 80:
    return df.clean_institution
  else:
      return df.institute_match



if __name__ == '__main__':

    os.chdir('/Users/florencehope/OpenData/')
    aff_matched = pd.read_csv('matched_studies_25stop_small.csv')
    study_metadata = pd.read_csv('data/archive/metadata.csv')

    affs = aff_matched[['paper_id','author_number','clean_institution','institute_match','institute_match_country','match_ratio']]

    affs['institute'] = affs.apply(institution_label,axis=1)
    #affs['country'] = affs.apply(country_label, axis = 1)


    grpAffsInstitute = group_by_key(affs,'institute')

    #grpAffsCountry = group_by_key(affs,'country')

    studies = study_metadata.merge(grpAffsInstitute,left_on='paper_id',right_on = 'sha') #why losing from this join?
    #studies = studies.merge(grpAffsCountry,left_on='paper_id',right_on = 'sha') #why losing from this join?

    studies.to_csv('studies_with_meta.csv')

    daily = studies.groupby('publish_time').agg({'collaborative_ind':'sum','paper_id':'count','num_institutions':'mean'}).reset_index()


    daily = daily[daily['publish_time'] >= '2020-03-01']
    daily =  daily[daily['publish_time'] <= '2021-01-01']

    daily['collab_rate'] = daily['collaborative_ind']/daily['paper_id']

    daily['rolling_collab'] = daily['collab_rate'].rolling(14, center=True).mean()
    daily['rolling_papers'] = daily['paper_id'].rolling(14, center=True).mean()


    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('day')
    ax1.plot( daily.rolling_collab, color='red')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(daily.rolling_papers, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    daily.collaborative_ind.plot()
