import argparse
import numpy as np
import pandas as pd
from skbio import TreeNode
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K
from LossFunctionMethods import weighted_unifrac
from UnifracLoss import weighted_unifrac_loss

data_tree = TreeNode.read('/Users/remycross/Desktop/WXS to WGS/species.rooted.nw')
taxa = pd.read_table('/Users/remycross/Downloads/taxonomy.txt', index_col=0)
metadata = pd.read_table('/Users/remycross/Downloads/metadata.WGS.solid.sample.txt', index_col=0)


def create_data():
    projects = ['COAD', 'READ', 'HNSC', 'STAD', 'ESCA']
    d = {}
    for assay in ['WGS', 'WXS']:
        dframes = []
        for project in projects:
            fpath = '/Users/remycross/Downloads/remy_data/{}/{}/'.format(project, assay)
            fname = 'bacteria.unambiguous.decontam.tissue.sample.reads.txt'
            df = pd.read_table(fpath + fname, index_col=0).transpose()
            dframes.append(df)
        d[assay] = pd.concat(dframes).fillna(0.0)

    wgs = d['WGS']
    wxs = d['WXS']
    ix = np.intersect1d(wgs.index, wxs.index)
    wgs = wgs.loc[ix]
    wxs = wxs.loc[ix]

    return wxs, wgs


def get_taxa(tax):
    taxon = tax
    valid_tree_names = []
    count = 0
    for node in data_tree.postorder():
        if node.name != 'root':
            count += 1
            valid_tree_names.append(int(node.name))
    valid = list(set(valid_tree_names))
    keep_taxa = np.intersect1d(wxs.columns, valid)
    taxid = taxa.index[taxa.type == taxon]
    keep_taxa = np.intersect1d(keep_taxa, taxid)
    return keep_taxa


def nonzero_reads(data):
    counts = []
    for i in range(len(data.index)):
        counts.append(len(np.nonzero(data.iloc[i].to_numpy() > 1)[0]))
    counts = dict(zip(data.index, counts))
    return counts


def filter_taxa(x, y, threshold):
    wgs_prev = nonzero_reads(y.T)
    wgs_prev = dict(zip(wgs_prev.keys(), [j / len(y.index) for j in wgs_prev.values()]))
    print(wgs_prev.values())
    # for i in wgs_prev.keys():
    #     print(wgs_prev[i])
    wgs_keep_taxa = [i for i in wgs_prev.keys() if wgs_prev[i] >= threshold]
    taxa = np.intersect1d(x.columns, wgs_keep_taxa)
    empty_otus_wxs = np.setdiff1d(wgs_keep_taxa, x.columns)
    wxs_empty = pd.DataFrame(index=x.index, columns=empty_otus_wxs).fillna(0.0)
    y = y[wgs_keep_taxa]
    x = pd.concat([x[taxa], wxs_empty], axis=1)[wgs_keep_taxa]
    wxs = np.log10(x[wgs_keep_taxa].fillna(0.0) + 0.1) + 1  # WXS
    wgs = np.log10(y[wgs_keep_taxa].fillna(0.0) + 0.1) + 1  # WGS
    return wxs, wgs


def filter_samples(x, y, threshold):
    wgs_abund = np.log10(y.sum(axis=1) + .1) + 1
    samples = np.where(wgs_abund >= threshold)
    return samples


def run_neural_network(x, y, s_thresh, t_thresh):
    wxs_train, wxs_test, wgs_train, wgs_test = train_test_split(x, y, random_state=1)
    model = Sequential()
    model.add(Dense((len(x.columns) * 1.5), input_dim=len(x.columns)))
    model.add(Dense(len(x.columns)))
    model.compile(loss=weighted_unifrac_loss, optimizer='adam', metrics=['mse'])
    history = model.fit(wxs_train.to_numpy(), wgs_train.to_numpy(), epochs=1, batch_size=20)
    wgs_pred = model.predict(wxs_test.to_numpy())
    unidist = []
    for i in range(len(wgs_pred)):
        unidist.append(
            K.square(weighted_unifrac(wgs_pred[i], wgs_test.to_numpy()[i], otu_ids, data_tree, validate=False,
                                      normalized=True)))
    unidist = pd.DataFrame(unidist, columns=[0])
    wgs_pred = pd.DataFrame(wgs_pred, index=wxs_test.index, columns=wxs_test.columns)
    wgs_pred.to_csv(
        '/Users/remycross/Desktop/WXS to WGS/KerasModel/ThresholdsCSV/Predicted_{}_{}.csv'.format(s_thresh, t_thresh))
    unidist.to_csv(
        '/Users/remycross/Desktop/WXS to WGS/KerasModel/ThresholdsCSV/Unifrac_{}_{}.csv'.format(s_thresh, t_thresh))


def create_parser():
    parser = argparse.ArgumentParser(
        description="""Creates a t by s sized matrix where the value is the unifrac distance testing accuracy
                    when the dataset uses specifications t and s""")

    parser.add_argument('-t', '--taxa-threshold', nargs='?', required=True,
                        help="""Threshold for taxa prevalence (eg. 0.01)""")
    parser.add_argument('-s', '--sample-threshold', nargs='?', required=True,
                        help="""Threshold for log-normalized abundance of samples (eg. 4)""")
    return parser


def weighted_unifrac_loss(y_true, y_pred):
    sums = 0.0
    for i in range(len(y_true)):
        sums += float(
            K.square(weighted_unifrac(y_true[i], y_pred[i], otu_ids, data_tree, validate=False, normalized=True)))

    return sums / len(y_true)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    d = vars(args)
    sample_threshold = d['sample_threshold']
    taxa_threshold = d['taxa_threshold']
    wxs, wgs = create_data()
    taxa_list = wxs.columns
    samples = filter_samples(wxs, wgs, float(sample_threshold))
    wxs, wgs = filter_taxa(wxs, wgs, float(taxa_threshold))
    keep = get_taxa('species')
    wxs = wxs[keep].iloc[samples]
    wgs = wgs[keep].iloc[samples]
    otu_ids = [str(x) for x in wgs.columns]
    run_neural_network(wxs, wgs, sample_threshold, taxa_threshold)
