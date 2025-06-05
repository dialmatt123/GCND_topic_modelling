import pandas as pd
import numpy as np

# ngram = 'fourgram'
# components = '5'
ngram_list = ['trigram','fourgram','words']
comp_list = [2, 3, 4, 5]

def loc_incoherence(ngram, components):
    # import classification file
    grp = pd.read_csv(f"./GCND_full/nearest_neighbour/data/nmf_{ngram}_gcnd_full_modified_cleaned_{components}_1.0_relevanceFalse_idfTrue_norml2_subTrue.csv", index_col = 0)
    # subset kloeke codes and dominant dialect groups
    grp = grp[['DocID','dominant_topic']]
    # dictionary for modern tone category per variety for the specific word
    grp_dict = dict(zip(grp['DocID'], grp['dominant_topic']))

    # define number of neighbours
    num_neighbours = 10
    # import nearest neighbours
    # To get nearest neighbours, you are required to obtain these neighbours either on QGIS or by other means
    nn = pd.DataFrame(pd.read_csv(f"./GCND_full/nearest_neighbour/{num_neighbours}NN.csv"))

    # reference dialect dominant group
    ref_grp = []
    # get tones for reference dialects
    for i in range(len(nn)):
        if nn.iloc[i,0] not in grp_dict:
            ref_grp.append('nan')
        else:
            ref_grp.append(grp_dict.get(nn.iloc[i,0]))

    # target dialect dominant group
    tar_grp = []
    # get tones for reference dialects
    for i in range(len(nn)):
        if nn.iloc[i,1] not in grp_dict:
            tar_grp.append('nan')
        else:
            tar_grp.append(grp_dict.get(nn.iloc[i,1]))

    # add the tone values to the df
    nn['Reference Group'] = ref_grp
    nn['Target Group'] = tar_grp

    pair_incoh = []
    # Pairwise incoherence: incoherence between ref dialect and a neighbour
    for i in range(len(nn)):
        # if the neighbour does not match the tone reflex category of the reference dialect
        if nn.iloc[i,2] == 'nan' or nn.iloc[i,3] == 'nan':
            pair_incoh.append(np.nan)
        elif nn.iloc[i,2] != nn.iloc[i,3]:
            pair_incoh.append(1)
        else:
            pair_incoh.append(0.000000001)

    nn['Incoherence'] = pair_incoh
    # Calculate the average of every N rows in Column1
    local_incoh = pd.DataFrame(nn.groupby(nn.index // num_neighbours)['Incoherence'].apply(lambda x: round(x.dropna().mean(),2)))

    # coord is for plotting the topic model results on QGIS
    coord = pd.read_csv(f"./GCND_full/nearest_neighbour/data/GCND_full_modified_cleaned_NMF_{ngram}_{components}_components_QGIS.csv", index_col=0)
    coord = coord[['DocID','latitude','longitude']]
    coord = pd.concat([coord,local_incoh], axis=1)
    coord.to_csv(f"./GCND_full/nearest_neighbour/data/GCND_full_modified_cleaned_NMF_{ngram}_{components}_components_LocIncol_QGIS.csv")
    print(ngram, str(components)+"components", "\nMean Local Incoherence:\t\t", coord["Incoherence"].mean())
    print('file exported.')

for n in ngram_list:
    for component in comp_list:
        loc_incoherence(n, component)
