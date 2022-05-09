import argparse
import numpy as np
import pandas as pd
from skbio import TreeNode
from sklearn.model_selection import train_test_split
from tensorflow import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from Thresholds import filter_samples, filter_taxa, get_taxa
from LossFunctionMethods import weighted_unifrac

# from UnifracLoss import weighted_unifrac_loss

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


def run_neural_network(x, y, hidden, str):
    wxs_train, wxs_test, wgs_train, wgs_test = train_test_split(x, y, random_state=1)
    model = Sequential()
    for i, n in enumerate(hidden):
        if i == 0:
            model.add(Dense(n* len(x.columns), input_dim=len(x.columns)))
        else:
            model.add(Dense(n * len(x.columns)))
    model.add(Dense(len(x.columns)))
    model.compile(loss=weighted_unifrac_loss, optimizer='adam', metrics=['mse'])
    history = model.fit(wxs_train.to_numpy(), wgs_train.to_numpy(), epochs=1, batch_size=20)
    wgs_pred = model.predict(wxs_test.to_numpy())
    print(type(wgs_pred))
    unidist = []
    for i in range(len(wgs_pred)):
        print(K.square(weighted_unifrac(wgs_pred[i], wgs_test.to_numpy()[i], otu_ids, data_tree, validate=False,
                                        normalized=True)))
        unidist.append(
            K.square(weighted_unifrac(wgs_pred[i], wgs_test.to_numpy()[i], otu_ids, data_tree, validate=False,
                                      normalized=True)))
    unidist = pd.DataFrame(unidist, columns=[0])
    wgs_pred = pd.DataFrame(wgs_pred, index=wxs_test.index, columns=wxs_test.columns)
    wgs_pred.to_csv('/Users/remycross/Desktop/WXS to WGS/KerasModel/ArchitecturesCSV/Predicted_{}Layers.csv'.format(str))
    unidist.to_csv(
        '/Users/remycross/Desktop/WXS to WGS/KerasModel/ArchitecturesCSV/Unifrac_{}Layers.csv'.format(str))


def create_parser():
    parser = argparse.ArgumentParser(
        description="""Creates a t by s sized matrix where the value is the unifrac distance testing accuracy
                    when the dataset uses specifications t and s""")

    parser.add_argument('-l', '--layers', nargs='?', required=True,
                        help="""Hidden layers in the ANN (eg. 1_1.5_1 for 3 hidden layers with value*columns nodes)""")

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
    layers_string = d['layers']
    layers = [int(x) for x in d['layers'].split('_')]
    print(layers)
    wxs, wgs = create_data()
    samples = filter_samples(wxs, wgs, float(3))
    wxs, wgs = filter_taxa(wxs, wgs, float(.005))
    print(type(wxs))
    keep = get_taxa('species')
    wxs = wxs[keep].iloc[samples]
    wgs = wgs[keep].iloc[samples]
    otu_ids = [str(x) for x in wgs.columns]
    run_neural_network(wxs, wgs, layers, layers_string)
