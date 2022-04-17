import colorsys, datetime, os, pathlib, platform, pprint, sys, contextlib
import fastai
#import gtda
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import PIL.Image as Image
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns
#import sdv
import sklearn
import torch
import torch.nn as nn
import yellowbrick as yb
import imblearn 
from imblearn.over_sampling import SMOTE, SMOTENC
from collections import Counter
from matplotlib import pyplot

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline                             
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV    

import pickle
from collections import namedtuple

from fastai.tabular.all import FillMissing, Categorify, Normalize, tabular_learner, accuracy, ClassificationInterpretation, ShowGraphCallback, RandomSplitter, range_of
from fastai.callback.all import CollectDataCallback, Callback, ProgressCallback
from fastai.callback.all import LRFinder, EarlyStoppingCallback, SaveModelCallback, ShowGraphCallback, CSVLogger, slide, steep, minimum, valley
from fastai.data.all import Transform, DisplayedTransform
from fastai.metrics import MatthewsCorrCoef, F1Score, Recall, Precision, RocAuc, BalancedAccuracy
from fastai.optimizer import ranger, Adam
from fastai.tabular.data import TabularDataLoaders, TabularPandas, MultiCategoryBlock
from fastai.tabular.all import FillMissing, Categorify, Normalize, tabular_learner, accuracy, ClassificationInterpretation, range_of
from fastai.tabular.all import get_emb_sz, Module, Learner, Embedding, CrossEntropyLossFlat, IndexSplitter, ColSplitter, RandomSplitter
from fastai.tabular.all import delegates, tabular_config, TabularLearner, get_c, ifnone, is_listy, LinBnDrop, SigmoidRange 
from fastai.test_utils import synth_learner, VerboseCallback
from fastcore.all import Transform, store_attr, L
from fastcore.all import L
from fastai.basics import Tensor


from fast_tabnet.core import TabNetModel as TabNet, tabnet_feature_importances, tabnet_explain

from fastcore.all import Transform, store_attr

fastai_gpu_conflict = True
if(fastai_gpu_conflict):
    torch.cuda.is_available = lambda : False

import copy

#from gtda.curves import Derivative, StandardFeatures
#from gtda.diagrams import PersistenceEntropy, Amplitude, NumberOfPoints, ComplexPolynomial, PersistenceLandscape, PersistenceImage, BettiCurve, Silhouette, HeatKernel
#from gtda.homology import VietorisRipsPersistence, EuclideanCechPersistence
#from gtda.plotting import plot_diagram, plot_point_cloud, plot_betti_curves, plot_betti_surfaces

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from pytorch_tabnet.tab_network import TabNetNoEmbeddings

#from sdv.tabular import TVAE, CopulaGAN, CTGAN, GaussianCopula
#from sdv.evaluation import evaluate

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from yellowbrick.model_selection import CVScores, LearningCurve, ValidationCurve

seed: int = 14

# allow plots to be shown in the notebook
#init_notebook_mode(connected=True)
# pio.renderers.default = "iframe"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set up pandas display options
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

# set up pretty printer for easier data evaluation
pretty = pprint.PrettyPrinter(indent=4, width=30).pprint


# print library and python versions for reproducibility
print(
    f'''
    Last Execution: {datetime.datetime.now()}
    python:\t{platform.python_version()}

    \tfastai:\t\t{fastai.__version__}
    \tmatplotlib:\t{mpl.__version__}
    \tnumpy:\t\t{np.__version__}
    \tpandas:\t\t{pd.__version__}
    \tseaborn:\t{sns.__version__}
    \tsklearn:\t{sklearn.__version__}
    \ttorch:\t\t{torch.__version__}
    \tyellowbrick:\t{yb.__version__}
    \timblearn:\t{imblearn.__version__}
    '''
)

def get_file_path(directory: str):
    '''
        Closure that will return a function. 
        Function will return the filepath to the directory given to the closure
    '''

    def func(file: str) -> str:
        return os.path.join(directory, file)

    return func



def load_data(filePath):
    '''
        Loads the Dataset from the given filepath and caches it for quick access in the future
        Function will only work when filepath is a .csv file
    '''

    p = pathlib.Path(filePath)
    filePathClean: str = str(p.parts[-1])
    pickleDump: str = f'./cache/{filePathClean}.pickle'

    print(f'Loading Dataset: {filePath}')
    print(f'\tTo Dataset Cache: {pickleDump}\n')


    # check if data already exists within cache
    if os.path.exists(pickleDump):
        df = pd.read_pickle(pickleDump)
 
    # if not, load data and clean it before caching it
    else:
        df = pd.read_csv(filePath, low_memory=True)
        df.to_pickle(pickleDump)
    
    return df



def features_with_bad_values(df: pd.DataFrame, datasetName: str) -> pd.DataFrame:
    '''
        Function will scan the dataframe for features with Inf, NaN, or Zero values.
        Returns a new dataframe describing the distribution of these values in the original dataframe
    '''

    # Inf and NaN values can take different forms so we screen for every one of them
    invalid_values: list = [ np.inf, np.nan, 'Infinity', 'inf', 'NaN', 'nan', 0 ]
    infs          : list = [ np.inf, 'Infinity', 'inf' ]
    NaNs          : list = [ np.nan, 'NaN', 'nan' ]

    # We will collect stats on the dataset, specifically how many instances of Infs, NaNs, and 0s are present.
    # using a dictionary that will be converted into a (3, n) dataframe where n is the number of features in the dataset
    stats: dict = {
        'Dataset':[ datasetName, datasetName, datasetName ],
        'Value'  :['Inf', 'NaN', 'Zero']
    }

    i = 0
    for col in df.columns:
        
        i += 1
        feature = np.zeros(3)
        
        for value in invalid_values:
            if value in infs:
                j = 0
            elif value in NaNs:
                j = 1
            else:
                j = 2
            indexNames = df[df[col] == value].index
            if not indexNames.empty:
                feature[j] += len(indexNames)
                
        stats[col] = feature

    return pd.DataFrame(stats)



def clean_data(df: pd.DataFrame, prune: list) -> pd.DataFrame:
    '''
        Function will take a dataframe and remove the columns that match a value in prune 
        Inf and Nan values will also be removed once appropriate rows and columns 
        have been removed, 
        we will return the dataframe with the appropriate values
    '''

    # remove the features in the prune list    
    for col in prune:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    
    # drop missing values/NaN etc.
    df.dropna(inplace=True)

    
    # Search through dataframe for any Infinite or NaN values in various forms that were not picked up previously
    invalid_values: list = [
        np.inf, np.nan, 'Infinity', 'inf', 'NaN', 'nan'
    ]
    
    for col in df.columns:
        for value in invalid_values:
            indexNames = df[df[col] == value].index
            if not indexNames.empty:
                print(f'deleting {len(indexNames)} rows with Infinity in column {col}')
                df.drop(indexNames, inplace=True)

    return df


color_list: list = [
    'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
    'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
    'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
    'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
    'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
    'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
    'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
    'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
    'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
    'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
    'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
    'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
    'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
    'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
    'ylorrd'
]


def get_n_color_list(n: int, opacity=.8, luminosity=.5, saturation=.5) -> list:
    '''
        Function creates a list of n distinct colors, formatted for use in a plotly graph
    '''

    colors = []

    for i in range(n):
        color = colorsys.hls_to_rgb(
                i / n,
                luminosity,
                saturation,
        )
        
        colors.append(f'rgba({color[0] * 255}, {color[1] * 255}, {color[2] * 255}, {opacity})')

    return colors


# Sklearn classes

class SklearnWrapper(BaseEstimator):
    '''
        A wrapper for fastai learners for creating visualizations using yellowbrick
        code sourced from: 
        forums.fast.ai/t/fastai-with-yellowbrics-how-to-get-roc-curves-more/79408
    '''
    _estimator_type = "classifier"
        
    def __init__(self, model):
        self.model = model
        self.classes_ = list(self.model.dls.y.unique())
        self._calls = [] 
    
    def fit(self, X, y):
        self._calls.append(('fit', X, y))
        pass
        
    def score(self, X, y):
        self._calls.append(('score', X, y))
        return accuracy_score(y, self.predict(X))
    
    def get_new_preds(self, X):
        self._calls.append(('get_new_preds', X))
        new_to = self.model.dls.valid_ds.new(X)
        new_to.conts = new_to.conts.astype(np.float32)
        new_dl = self.model.dls.valid.new(new_to)
        with self.model.no_bar():
            preds,_,dec_preds = self.model.get_preds(dl=new_dl, with_decoded=True)
        return (preds, dec_preds)

    def predict_proba(self, X):
        self._calls.append(('predict_proba', X))
        return self.get_new_preds(X)[0].numpy()
    
    def predict(self, X):
        self._calls.append(('predict', X))
        return self.get_new_preds(X)[1].numpy()


# fastai classes



class PCA_tabular(Transform):
    '''
        Class will implement a PCA feature extraction method for tabular data
        On setup, we train a pca on the training data, then extract n_comps from the entire dataset 
            the components are then added to the dataframe as new columns
    '''
    def __init__(self, n_comps=3, add_col=True):
        store_attr()

    def setups(self, to, **kwargs):

        self.pca = PCA(n_components=self.n_comps)
        self.pca.fit(to.train.conts)
        pca = pd.DataFrame(self.pca.transform(to.conts))
        pca.columns = [f'pca_{i+1}' for i in range(self.n_comps)]

        for col in pca.columns:
            to.items[col] = pca[col].values.astype('float32')
        

        if self.add_col:
            for i in range(self.n_comps):
                if f'pca_{i+1}' not in to.cont_names: to.cont_names.append(f'pca_{i+1}')
        
        return self(to)



class Normal(DisplayedTransform):
    '''
        A data processing tool inherited from the fastai library. 
        This is a modified version of the normalize function that performs MinMax scaling and is able to be
            used in our preprocessing pipeline. 
        The original normalizes the data to have a mean centered at 0 and a standard deviation of 1.
    '''
    def setups(self, to, **kwargs):
        self.mins = getattr(to, 'train', to).conts.min()
        self.maxs = getattr(to, 'train', to).conts.max()
        
        return self(to)
        
    def encodes(self, to, **kwargs):
        to.conts = (to.conts-self.mins) / (self.maxs - self.mins)
        return to

    def decodes(self, to, **kwargs):
        to.conts = (to.conts) * (to.maxs - to.mins) + to.mins
        return to


def examine_dataset(job_id: int) -> dict:
    '''
        Function will return a dictionary containing dataframe of the job_id passed in as well as that dataframe's
        feature stats, data composition, and file name.

        This dictionary is expected as the input for all of the other helper functions
    '''

    job_id = job_id - 1  # adjusts for indexing while enumerating jobs from 1
    print(f'Dataset {job_id+1}/{len(data_set)}: We now look at {file_set[job_id]}\n\n')

    # Load the dataset
    df: pd.DataFrame = load_data(file_set[job_id])
 

    # print the data composition
    print(f'''
        File:\t\t\t\t{file_set[job_id]}  
        Job Number:\t\t\t{job_id+1}
        Shape:\t\t\t\t{df.shape}
        Samples:\t\t\t{df.shape[0]} 
        Features:\t\t\t{df.shape[1]}
    ''')
    

    # return the dataframe and the feature stats
    data_summary: dict =  {
        'File':             file_set[job_id],
        'Dataset':          df,
        'Feature_stats':    features_with_bad_values(df, file_set[job_id]), 
    }
    
    return data_summary



def package_data_for_inspection(df: pd.DataFrame) -> dict:
    '''
        Function will return a dictionary containing dataframe passed in as well as that dataframe's feature stats.
    '''

    # print the data composition
    print(f'''
    Dataset statistics:
        Shape:\t\t\t\t{df.shape}
        Samples:\t\t\t{df.shape[0]} 
        Features:\t\t\t{df.shape[1]}
    ''')
    

    # return the dataframe and the feature stats
    data_summary: dict =  {
        'File':             '',
        'Dataset':          df,
        'Feature_stats':    features_with_bad_values(df, ''), 
    }
    
    return data_summary



def package_data_for_inspection_with_label(df: pd.DataFrame, label: str) -> dict:
    '''
        Function will return a dictionary containing dataframe passed in as well as that dataframe's feature stats.
    '''

    # print the data composition
    print(f'''
        Shape:\t\t\t\t{df.shape}
        Samples:\t\t\t{df.shape[0]} 
        Features:\t\t\t{df.shape[1]}
    ''')
    

    # return the dataframe and the feature stats
    data_summary: dict =  {
        'File':             f'{label}',
        'Dataset':          df,
        'Feature_stats':    features_with_bad_values(df, f'{label}'),
    }
    
    return data_summary



def check_infs(data_summary: dict) -> pd.DataFrame:
    '''
        Function will return a dataframe of features with a value of Inf.
    '''

    
    vals: pd.DataFrame = data_summary['Feature_stats']
    inf_df = vals[vals['Value'] == 'Inf'].T

    return inf_df[inf_df[0] != 0]



def check_nans(data_summary: dict) -> pd.DataFrame:
    '''
        Function will return a dataframe of features with a value of NaN.
    '''

    vals: pd.DataFrame = data_summary['Feature_stats']
    nan_df = vals[vals['Value'] == 'NaN'].T

    return nan_df[nan_df[1] != 0]



def check_zeros(data_summary: dict) -> pd.DataFrame:
    '''
        Function will return a dataframe of features with a value of 0.
    '''

    vals: pd.DataFrame = data_summary['Feature_stats']
    zero_df = vals[vals['Value'] == 'Zero'].T

    return zero_df[zero_df[2] != 0]



def check_zeros_over_threshold(data_summary: dict, threshold: int) -> pd.DataFrame:
    '''
        Function will return a dataframe of features with a value of 0.
    '''

    vals: pd.DataFrame = data_summary['Feature_stats']
    zero_df = vals[vals['Value'] == 'Zero'].T
    zero_df_bottom = zero_df[2:]

    return zero_df_bottom[zero_df_bottom[2] > threshold]



def check_zeros_over_threshold_percentage(data_summary: dict, threshold: float) -> pd.DataFrame:
    '''
        Function will return a dataframe of features with all features with
        a frequency of 0 values greater than the threshold
    '''

    vals: pd.DataFrame = data_summary['Feature_stats']
    size: int = data_summary['Dataset'].shape[0]
    zero_df = vals[vals['Value'] == 'Zero'].T
    zero_df_bottom = zero_df[2:]

    return zero_df_bottom[zero_df_bottom[2] > threshold*size]



def remove_infs_and_nans(data_summary: dict) -> pd.DataFrame:
    '''
        Function will return the dataset with all inf and nan values removed.
    '''

    df: pd.DataFrame = data_summary['Dataset'].copy()
    df = clean_data(df, [])

    return df



def rename_columns(data_summary: dict, columns: list, new_names: list) -> dict:
    '''
        Function will return the data_summary dict with the names of the columns in the dataframe changed
    '''

    df: pd.DataFrame = data_summary['Dataset'].copy()
    for x, i in enumerate(columns):
        df.rename(columns={i: new_names[x]}, inplace=True)

    data_summary['Dataset'] = df

    return data_summary



def rename_values_in_column(data_summary: dict, replace: list) -> pd.DataFrame:
    '''
        Function will return a dataframe with the names of the columns changed

        replace: [('column', {'old_name': 'new_name', ...}), ...]
    '''
    length: int = len(replace)

    df: pd.DataFrame = data_summary['Dataset'].copy()
    for i in range(length):
        df[replace[i][0]].replace(replace[i][1], inplace=True)


    return df



def rename_values_in_column_df(df: pd.DataFrame, replace: list) -> pd.DataFrame:
    '''
        Function will return a dataframe with the names of the columns changed

        replace: [('column', {'old_name': 'new_name', ...}), ...]
    '''
    length: int = len(replace)

    df1: pd.DataFrame = df.copy()
    for i in range(length):
        df1[replace[i][0]].replace(replace[i][1], inplace=True)


    return df1



def prune_dataset(data_summary: dict, prune: list) -> pd.DataFrame:
    '''
        Function will return the dataset with all the columns in the prune list removed.
    '''

    df: pd.DataFrame = data_summary['Dataset'].copy()
    df = clean_data(df, prune)

    return df



def prune_feature_by_values(df: pd.DataFrame, column: str, value: list) -> pd.DataFrame:
    '''
        Function takes a dataframe, a column name, and a list of values and returns a dataframe
        with all rows that do not have the values in the column removed

        Deprecated, use reduce_feature_to_values:
            ambiguous name implied it is removing values, not keeping them
    '''
    new_df = pd.DataFrame()
    for v in value:
        new_df = new_df.append(df[df[column] == v].copy())

    return new_df



def reduce_feature_to_values(df: pd.DataFrame, column: str, value: list) -> pd.DataFrame:
    '''
        Function takes a dataframe, a column name, and a list of values and returns a dataframe
        with all rows that do not have the values in the column removed
    '''
    new_df = pd.DataFrame()
    for v in value:
        new_df = new_df.append(df[df[column] == v].copy())

    return new_df



def test_infs(data_summary: dict) -> bool:
    '''
        Function asserts the dataset has no inf values.
    '''
    vals: pd.DataFrame = data_summary['Feature_stats']
    inf_df = vals[vals['Value'] == 'Inf'].T

    assert inf_df[inf_df[0] != 0].shape[0] == 2, 'Dataset has inf values'
    

    return True



def test_nans(data_summary: dict) -> bool:
    '''
        Function asserts the dataset has no NaN values
    '''

    vals: pd.DataFrame = data_summary['Feature_stats']
    nan_df = vals[vals['Value'] == 'NaN'].T

    assert nan_df[nan_df[1] != 0].shape[0] == 2, 'Dataset has NaN values'


    return True



def test_pruned(data_summary: dict, prune: list) -> bool:
    '''
        Function asserts the dataset has none of the columns present in the prune list 
    '''

    pruned: bool = True

    for col in prune:
        if col in data_summary['Dataset'].columns:
            pruned = False

    assert pruned, 'Dataset has columns present in prune list'

    return pruned



def test_pruned_size(data_summary_original: dict, data_summary_pruned: dict, prune: list) -> bool:
    '''
        Function asserts the dataset has none of the columns present in the prune list 
    '''

    original_size: int = data_summary_original['Dataset'].shape[1]
    pruned_size: int = data_summary_pruned['Dataset'].shape[1]
    prune_list_size: int = len(prune)

    assert original_size - prune_list_size == pruned_size, 'Dataset has columns present in prune list'

    return True

class DFLogger(Callback):
    '''
        Class defines a callback that is passed to the fastai learner that
            will save the recorded metrics for each epoch to a dataframe
    '''
    order=60
    def __init__(self, fname='history.csv', append=False):
        self.fname,self.append = pathlib.Path(fname),append
        self.df = pd.DataFrame()
        self.flag = True

    def to_csv(self, file: str or None = None):
        if file is None:
            self.df.to_csv(self.path/self.fname, index=False)
        else:
            self.df.to_csv(file, index=False)

    def before_fit(self):
        "Prepare file with metric names."
        if hasattr(self, "gather_preds"): return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.old_logger,self.learn.logger = self.logger,self.add_row

    def add_row(self, log):
        "Write a line with `log` and call the old logger."
        if(self.flag):
            self.df = pd.DataFrame([log], columns=self.recorder.metric_names)
            self.flag = False
        else:
            if (len(log) == len(self.df.columns)):
                self.new_row = pd.DataFrame([log], columns=self.recorder.metric_names)
                self.df = pd.concat([self.df, self.new_row], ignore_index=True)

    def after_fit(self):
        "Close the file and clean up."
        if hasattr(self, "gather_preds"): return
        self.learn.logger = self.old_logger



class LazyGraphCallback(Callback):
    '''
        Class defines a callback that is passed to the fastai learner that
            saves the validation and training loss metrics to graph when
            calling the .plot_graph() method
        
        This allows us to display the train/validation loss graph after the    
            experiment is run, even if it is run in no_bar mode
    '''
    order: int = 65
    run_valid: bool = False
    
    def __init__(self):
        self.graph_params: list = []
        self.graphs: list = []

    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds")
        if not self.run: return
        self.nb_batches: list = []

    def after_train(self): self.nb_batches.append(self.train_iter)

    def after_epoch(self):
        "Plot validation loss in the pbar graph"
        if not self.nb_batches: return
        rec = self.learn.recorder
        iters = range_of(rec.losses)
        val_losses = [v[1] for v in rec.values]
        x_bounds = (0, (self.n_epoch - len(self.nb_batches)) * self.nb_batches[0] + len(rec.losses))
        y_bounds = (0, max((max(Tensor(rec.losses)), max(Tensor(val_losses)))))
        self.graph_params.append(([(iters, rec.losses), (self.nb_batches, val_losses)], x_bounds, y_bounds))


    def plot_graph(self, ax: plt.Axes or None = None, title: str = 'Loss'):

        params = self.graph_params[-1]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(title)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.plot(params[0][0][0], params[0][0][1], label='Training')
        ax.plot(params[0][1][0], params[0][1][1], label='Validation')
        ax.legend()



class ModelStatsCallback(Callback):
    '''
        Collect all batches, along with `pred` and `loss`, into `self.datum`.
        This allows the data to be used after the model is used.
    '''
    order=60 # mysterious param we need to investigate
    def __init__(self):
        self.datum: list = []
        self.records: list = []
        self.c_idx: int  = -1

    def before_fit(self):
        self.datum.append(L())
        self.records.append({'loss': [], 'predictions': []})
        self.c_idx += 1


    def after_batch(self):
        vals = self.learn.to_detach((self.xb,self.yb,self.pred,self.loss))
        self.datum[self.c_idx].append(vals)
        self.records[self.c_idx]['predictions'].append(vals[2])
        self.records[self.c_idx]['loss'].append(vals[3])
        self.records[self.c_idx]['iters'] = range_of(self.recorder.losses)

## pytorch --> fastai models

class ResidualBlock(Module):
    def __init__(self, module, layer):
        # super().__init__()
        self.module = module
        self.layer = layer

    def forward(self, inputs):
        fx = self.module(inputs)
        if(inputs.shape != fx.shape):
            print('mismatch at layer:', self.layer ,inputs.shape, fx.shape)
        return fx + inputs


class ResidualTabularModel(Module):
    "Residual model for tabular data."
    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.,
                 y_range=None, use_bn=True, bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True),
                 lin_first=True):
        ps = ifnone(ps, [0]*len(layers))
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]

        # print(f'sizes', sizes)
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]
        
        _layers: list = []
        num_residuals = 0
        residual_locations = []
        enum_length = len(list(enumerate(zip(ps+[0.],actns))))
        for i, (p, a) in enumerate(zip(ps+[0.],actns)):
            # print(f'layer {i}: size i-1: {sizes[i-1]} idx={i-1}, size i: {sizes[i]} idx={i}, size i+1: {sizes[i+1]} idx={i+1}')
            if(i==0 or i == enum_length-1 or sizes[i] != sizes[i+1]):
            # if(i==0 or i == enum_length-1 or sizes[i] != sizes[i-1] or sizes[i] != sizes[i+1]):
                _layers.append(LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first))
            else:
                num_residuals += 1 
                residual_locations.append(i)
                _layers.append(
                    ResidualBlock(
                        LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first),
                        i
                    )
                )

        # _layers = [
        #     ResidualBlock(
        #         LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first)
        #     ) for i, (p,a) in enumerate(zip(ps+[0.],actns))
        # ]
        print(f'Layer sizes: {sizes}, length: {len(sizes)}')
        print(f'Number of residual blocks: {num_residuals}')
        print('Residual locations: ', residual_locations)

        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)



@delegates(Learner.__init__)
def residual_tabular_learner(dls, layers=None, emb_szs=None, config=None, n_out=None, y_range=None, **kwargs):
    "Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` created using the remaining params."
    if config is None: config = tabular_config()
    if layers is None: layers = [200,100]
    to = dls.train_ds
    emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = ResidualTabularModel(emb_szs, len(dls.cont_names), n_out, layers, y_range=y_range, **config)
    return TabularLearner(dls, model, **kwargs)


Model_data = namedtuple(
    'model_data', 
    ['name', 'model', 'classes', 'X_train', 'y_train', 'X_test', 'y_test', 'to', 'dls', 'model_type']
)

class ResidualBlock(Module):
    '''
        A simple residule block that creates a skip connection around any input module
        Output size of the input module must match the module's input size
    '''
    def __init__(self, module, layer):
        self.module = module
        self.layer = layer

    def forward(self, inputs):
        fx = self.module(inputs)
        if(inputs.shape != fx.shape):
            print('mismatch at layer:', self.layer ,inputs.shape, fx.shape)
            assert False
        return fx + inputs


class CardinalResidualBlock(Module):
    '''
        A residule block that creates a skip connection around a set of n branches
            where the number of branches is determined by the number of input modules
            in the branches list parameter.

        The output of the branches is summed together along with the input
        Output size of the input module must match the module's input size
    '''
    def __init__(self, branches: list, layer: int):
        self.branches = branches
        self.layer = layer

    def forward(self, inputs):
        fx = self.branches[0](inputs)
        if(inputs.shape != fx.shape):
            print('mismatch at layer:', self.layer ,inputs.shape, fx.shape)
            assert False
        if(len(self.branches) > 1):
            for i in range(len(self.branches) - 1):
                fx += self.branches[i + 1](inputs)

        return fx + inputs

# currently need to create a bottlenecking block that can be used to reduce the number of inputs
#   being passed by the residual connection 

class BottleneckResidualBlock(Module):
    '''
        A residule block that creates a skip connection around a set of n branches
            where the number of branches is determined by the number of input modules
            in the branches list parameter.

            the residual connection is put through a linear batchnormed layer if the
            input size is different from the output size
            Then, the output of the branches is summed together along with the possibly transformed input
    '''
    def __init__(self, branches: list, layer: int, in_size: int, out_size: int):
        self.branches = branches
        self.layer = layer

        self.in_size = in_size
        self.out_size = out_size

        # self.linear = nn.Linear(in_size, out_size)
        self.linear = LinBnDrop(in_size, out_size)

    def forward(self, inputs):

        fx = self.branches[0](inputs)
        for i in range(len(self.branches) - 1):
            fx += self.branches[i + 1](inputs)

        if(inputs.shape != fx.shape):
            inputs = self.linear(inputs)
        return fx + inputs


class ResidualTabularModel(Module):
    "Residual model for tabular data."
    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.,
                 y_range=None, use_bn=True, bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True),
                 lin_first=True, cardinality: list or None = None):
        ps = ifnone(ps, [0]*len(layers))
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]

        # print(f'sizes', sizes)
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]
        
        _layers: list = []
        num_residuals = 0
        residual_locations = []
        enum_length = len(list(enumerate(zip(ps+[0.],actns))))
        for i, (p, a) in enumerate(zip(ps+[0.],actns)):
            if(i==0 or i == enum_length-1):
                _layers.append(LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first))
            else:
                if(cardinality == None):
                    modules = [ LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first), ]
                else:
                    modules = [ LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first) for _ in range(cardinality[i]) ]
                num_residuals += 1 
                residual_locations.append(i)
                _layers.append( BottleneckResidualBlock(modules, i, sizes[i], sizes[i+1]) )

        print(f'Layer sizes: {sizes}, length: {len(sizes)}')
        print(f'Number of residual blocks: {num_residuals}')
        print('Residual locations: ', residual_locations)

        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)



@delegates(Learner.__init__)
def residual_tabular_learner(dls, layers=None, emb_szs=None, config=None, n_out=None, y_range=None, cardinality=None, ps=None, **kwargs):
    "Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` created using the remaining params."
    if config is None: config = tabular_config()
    if layers is None: layers = [200,100]
    to = dls.train_ds
    emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = ResidualTabularModel(emb_szs, len(dls.cont_names), n_out, layers, y_range=y_range, cardinality=cardinality, ps=ps, **config)
    return TabularLearner(dls, model, **kwargs)


def extract_validation_set(df: pd.DataFrame, target_label: str, categorical : list = ['Protocol'], leave_out: list = []):
    '''
        Run binary classification using a shallow learning model
        returns the 10-tuple Model_data
    '''
    dep_var: str = target_label

    categorical_features: list = []
    untouched_features  : list = []

    for x in categorical:
        if x in df.columns:
            categorical_features.append(x)

    for x in leave_out:
        if x in df.columns:
            untouched_features.append(x)
        
    continuous_features = list(set(df) - set(categorical_features) - set([dep_var]) - set(untouched_features))

    # Next, we set up the feature engineering pipeline, namely filling missing values
    # encoding categorical features, and normalizing the continuous features
    # all within a pipeline to prevent the normalization from leaking details
    # about the test sets through the normalized mapping of the training sets
    procs = [FillMissing, Categorify, Normalize]
    splits = RandomSplitter(valid_pct=.99, seed=seed)(range_of(df))
    
    # The dataframe is loaded into a fastai datastructure now that 
    # the feature engineering pipeline has been set up
    to = TabularPandas(
        df            , y_names=dep_var                , 
        splits=splits , cat_names=categorical_features ,
        procs=procs   , cont_names=continuous_features , 
    )

    # We use fastai to quickly extract the names of the classes as they are mapped to the encodings
    dls = to.dataloaders(bs=64)
    #model = utils.tabular_learner(dls)
    #classes : list = list(model.dls.vocab)


    # We extract the training and test datasets from the dataframe
    X_train = to.train.xs.reset_index(drop=True)
    X_test = to.valid.xs.reset_index(drop=True)
    y_train = to.train.ys.values.ravel()
    y_test = to.valid.ys.values.ravel()


    # We extract the training and test datasets from the dataframe
    X_test = to.valid.xs.reset_index(drop=True)
    y_test = to.valid.ys.values.ravel()
    
    return X_test, y_test

## Todo: make into general function that takes sklearn model instead of just a knn

def run_knn_experiment(df: pd.DataFrame, name: str, target_label: str, split=0.2, categorical : list = ['Protocol'], leave_out: list = []) -> Model_data:
    '''
        Run binary classification using K-Nearest Neighbors
        returns the 10-tuple Model_data
    '''

    # First we split the features into the dependent variable and 
    # continous and categorical features
    dep_var: str = target_label
    print(df.shape)

 
    categorical_features: list = []
    untouched_features  : list = []


    for x in categorical:
        if x in df.columns:
            categorical_features.append(x)

    for x in leave_out:
        if x in df.columns:
            untouched_features.append(x)
        
    continuous_features = list(set(df) - set(categorical_features) - set([dep_var]) - set(untouched_features))


    # Next, we set up the feature engineering pipeline, namely filling missing values
    # encoding categorical features, and normalizing the continuous features
    # all within a pipeline to prevent the normalization from leaking details
    # about the test sets through the normalized mapping of the training sets
    procs = [FillMissing, Categorify, Normalize]
    splits = RandomSplitter(valid_pct=split, seed=seed)(range_of(df))
    
    
    # The dataframe is loaded into a fastai datastructure now that 
    # the feature engineering pipeline has been set up
    to = TabularPandas(
        df            , y_names=dep_var                , 
        splits=splits , cat_names=categorical_features ,
        procs=procs   , cont_names=continuous_features , 
    )


    # We use fastai to quickly extract the names of the classes as they are mapped to the encodings
    dls = to.dataloaders(bs=64)
    model = tabular_learner(dls)
    classes : list = list(model.dls.vocab)


    # extract the name from the path
    p = pathlib.Path(name)
    name: str = str(p.parts[-1])


    # We extract the training and test datasets from the dataframe
    X_train = to.train.xs.reset_index(drop=True)
    X_test = to.valid.xs.reset_index(drop=True)
    y_train = to.train.ys.values.ravel()
    y_test = to.valid.ys.values.ravel()


    # Now that we have the train and test datasets, we set up a gridsearch of the K-NN classifier
    # using SciKitLearn and print the results 
    # params = {"n_neighbors": range(1, 50)}
    # model = GridSearchCV(KNeighborsClassifier(), params)
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    report = classification_report(y_test, prediction)
    print(report)
    print(f'\tAccuracy: {accuracy_score(y_test, prediction)}\n')
    # print("Best Parameters found by gridsearch:")
    # print(model.best_params_)


   # we add a target_type_ attribute to our model so yellowbrick knows how to make the visualizations
    if len(classes) == 2:
        model.target_type_ = 'binary'
    elif len(classes) > 2:  
        model.target_type_ = 'multiclass'
    else:
        print('Must be more than one class to perform classification')
        raise ValueError('Wrong number of classes')

    model_data: Model_data = Model_data(name, model, classes, X_train, y_train, X_test, y_test, to, dls, f'K_Nearest_Neighbors')

    # Now that the classifier has been created and trained, we pass out our training values
    # for analysis and further experimentation
    return model_data




def transform_and_split_data(df: pd.DataFrame, target_label: str, split=0, name='', categorical : list = ['Protocol'], scale:bool = False, leave_out: list = []) -> Model_data:
    '''
        Transform and split the data into a train and test set
        returns the 10-tuple with the following indicies:
        results: tuple = (name, model, classes, X_train, y_train, X_test, y_test, to, dls, model_type)
    '''

    # First we split the features into the dependent variable and 
    # continous and categorical features
    dep_var             : str  = target_label
    categorical_features: list = []
    untouched_features  : list = []

    print(df.shape)

    for x in categorical:
        if x in df.columns:
            categorical_features.append(x)

    for x in leave_out:
        if x in df.columns:
            untouched_features.append(x)

    continuous_features = list(set(df) - set(categorical_features) - set([dep_var]) - set(untouched_features))


    # Next, we set up the feature engineering pipeline, namely filling missing values
    # encoding categorical features, and normalizing the continuous features
    # all within a pipeline to prevent the normalization from leaking details
    # about the test sets through the normalized mapping of the training sets
    procs = [FillMissing, Categorify, Normalize]
    if(scale): 
        procs.append(Normal)

    splits = RandomSplitter(valid_pct=split, seed=seed)(range_of(df))
    
    
    # The dataframe is loaded into a fastai datastructure now that 
    # the feature engineering pipeline has been set up
    to = TabularPandas(
        df            , y_names=dep_var                , 
        splits=splits , cat_names=categorical_features ,
        procs=procs   , cont_names=continuous_features , 
    )


    # We use fastai to quickly extract the names of the classes as they are mapped to the encodings
    dls = to.dataloaders(bs=64)
    model = tabular_learner(dls)
    classes : list = list(model.dls.vocab)


    # We extract the training and test datasets from the dataframe
    X_train = to.train.xs.reset_index(drop=True)
    X_test = to.valid.xs.reset_index(drop=True)
    y_train = to.train.ys.values.ravel()
    y_test = to.valid.ys.values.ravel()


   # we add a target_type_ attribute to our model so yellowbrick knows how to make the visualizations
    if len(classes) == 2:
        model.target_type_ = 'binary'
    elif len(classes) > 2:  
        model.target_type_ = 'multiclass'
    else:
        model.target_type_ = 'single'
        
    model_data: Model_data = Model_data(name, model, classes, X_train, y_train, X_test, y_test, to, dls, 'Transformed_and_Split_data')

    # Now that the classifier has been created and trained, we pass out our training values
    # for analysis and further experimentation
    return model_data


Model_datum = namedtuple('Model_datum', ['models', 'results'])

# Todo: create crossvalidated experiments for resnet, tabnet, and models with a sklearn api



def run_cross_validated_deep_nn_experiment(
    df: pd.DataFrame, 
    name: str, 
    target_label: str, 
    shape: tuple, 
    k_folds: int = 10,
    categorical = ['Protocol'],
    dropout_rate: float = 0.01,
    callbacks: list = [ShowGraphCallback],
    metrics: list or None= None,
    procs: list = [FillMissing, Categorify, Normalize],
    experiment_type: str or None = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr_choice: str = 'valley'
) -> list:
    '''
        Function will fit a deep neural network to the given dataset using cross-validation
    '''
    metrics_ = metrics
    lr_choice = {
        'valley': 0,
        'slide': 1,
        'steep': 2,
        'minimum': 3,
    }[lr_choice]

    if(experiment_type is None):
        experiment_type = f'Deep_NN_{shape[0]}x{shape[1]}'

    p = pathlib.Path(name)
    name: str = str(p.parts[-1])
    dep_var: str = target_label

    print('Shape of input dataframe:', df.shape)
    print(f"Running {k_folds}-fold cross-validation")

    categorical_features: list = []
    untouched_features  : list = []


    for x in categorical:
        if x in df.columns:
            categorical_features.append(x)

        
    continuous_features = list(set(df) - set(categorical_features) - set([dep_var]) - set(untouched_features))



    ss = StratifiedShuffleSplit(n_splits=k_folds, random_state=seed, test_size=1/k_folds)


    model_data_list  : list = [0]*k_folds

    if metrics is None:
        metrics = [accuracy, BalancedAccuracy(), RocAuc(), MatthewsCorrCoef(), F1Score(average='macro'), Precision(average='macro'), Recall(average='macro')]
        fold_results: dict = {'loss': [], 'accuracy': [], 'BalancedAccuracy': [], 'roc_auc': [], 'MCC': [], 'f1': [], 'precision': [], 'recall': [], 'all': []}
    else:
        fold_results: dict = {'all': []}


    for i, (train_index, test_index) in enumerate(ss.split(df.copy().drop(target_label, axis=1), df[target_label])):
        
        fold_name = f'{name}_fold_{i+1}'

        print(i)
        print(f'Train Index: {train_index.shape}')
        print(f'Test Index: {test_index.shape}')

        splits = IndexSplitter(test_index)(df)
        to = TabularPandas(
            df            , y_names=dep_var                , 
            splits=splits , cat_names=categorical_features ,
            procs=procs   , cont_names=continuous_features , 
        )

        dls = to.dataloaders(bs=batch_size)

        model = tabular_learner(
            dls, 
            layers=list(shape), 
            metrics=metrics, 
            cbs=callbacks,
        )

        model.model.emb_drop = torch.nn.Dropout(p=dropout_rate)

        lr = model.lr_find(suggest_funcs=[valley, slide, steep, minimum])

        model.fit_one_cycle(epochs, lr[lr_choice])
        model.save(f'{name}.model')

        model_results = model.validate()

        if(metrics_ == None):
            fold_results['loss'].append(model_results[0])
            fold_results['accuracy'].append(model_results[1])
            fold_results['BalancedAccuracy'].append(model_results[2])
            fold_results['roc_auc'].append(model_results[3])
            fold_results['MCC'].append(model_results[4])
            fold_results['f1'].append(model_results[5])
            fold_results['precision'].append(model_results[6])
            fold_results['recall'].append(model_results[7])
            fold_results['all'].append(model_results)
            print(f'loss: {model_results[0]}, accuracy: {model_results[1]*100}%')
        else:
            fold_results['all'].append(model_results)
            print(f'loss: {model_results[0]}')

        X_train = to.train.xs.reset_index(drop=True)
        X_test = to.valid.xs.reset_index(drop=True)
        y_train = to.train.ys.values.ravel()
        y_test = to.valid.ys.values.ravel()

        wrapped_model = SklearnWrapper(model)
        wrapped_model._target_labels = dep_var
        classes = list(model.dls.vocab)



        # we add a target_type_ attribute to our model so yellowbrick knows how to make the visualizations
        if len(classes) == 2:
            wrapped_model.target_type_ = 'binary'
        elif len(classes) > 2:  
            wrapped_model.target_type_ = 'multiclass'
        else:
            print('Must be more than one class to perform classification')
            raise ValueError('Wrong number of classes')
        


        model_data: Model_data = Model_data(fold_name, wrapped_model, classes, X_train, y_train, X_test, y_test, to, dls, experiment_type)
        model_data_list[i] = model_data

    model_datum: Model_datum = Model_datum(model_data_list, fold_results)

    return model_datum

def run_residual_deep_nn_experiment(
    df: pd.DataFrame, 
    file_name: str, 
    target_label: str, 
    shape: tuple, 
    split=0.2, 
    categorical: list = ['Protocol'],
    procs = [FillMissing, Categorify, Normalize], 
    leave_out: list = [],
    epochs: int = 10,
    batch_size: int = 64,
    metrics: list or None = None,
    callbacks: list = [ShowGraphCallback],
    lr_choice: str = 'valley',
    name: str or None = None,
    fit_choice: str = 'one_cycle',
    cardinality: list or None = None,
    no_bar: bool = False,
    ps: list or None = None,
) -> Model_data:
    '''
        Function trains a residual deep neural network model on the given data. 
            Based on ResNet from Deep Residual Learning for Image Recognition by He et al. (2016) 
            as well as the ResNext network proposed by Xie et al. (2017) but adapted to tabular data  
        
        (https://arxiv.org/abs/1512.03385)
        (https://arxiv.org/abs/1611.05431)

        Parameters:
            df: pandas dataframe containing the data
            file_name: name of the file the dataset came from
            target_label: the label to predict
            shape: the shape of the neural network, the i-th value in the tuple represents the number of nodes in the i+1 layer
                    and the number of entries in the tuple represent the number of layers
            name: name of the experiment, if none a default is given
            split: the percentage of the data to use for testing
            categorical: list of the categorical columns
            procs: list of preprocessing functions to apply in the dataloaders pipeline
                    additional options are: 
                        PCA_tabular (generate n principal components) 
                        Normal (features are scaled to the interval [0,1])
            leave_out: list of columns to leave out of the experiment
            epochs: number of epochs to train for
            batch_size: number of samples processed in one forward and backward pass of the model
            metrics: list of metrics to calculate and display during training
            callbacks: list of callbacks to apply during training
            lr_choice: where the learning rate sampling function should find the optimal learning rate
                        choices are: 'valley', 'steep', 'slide', and 'minimum'
            fit_choice: choice of function that controls the learning schedule choices are:
                    'fit': a standard learning schedule 
                    'flat_cos': a learning schedule that starts high before annealing to a low value
                    'one_cyle': a learning schedule that warms up for the first epochs, continues at a high
                                    learning rate, and then cools down for the last epochs
                    the default is 'one_cycle'
            cardinality: list of integers that represent the number of residual blocks in each layer, if none
                            the default is one block per layer

        
        returns a model data named tuple
            model_data: tuple = (file_name, model, classes, X_train, y_train, X_test, y_test, model_type)
    '''
    shape = tuple(shape)

    if name is None:
        width: int = shape[0]
        for x in shape:
            width = x if (x > width) else width
        name = f'Residual_1D_Deep_NN_{len(shape)}x{width}'

    lr_choice = {'valley': 0, 'slide': 1, 'steep': 2, 'minimum': 3}[lr_choice]


    categorical_features: list = []
    untouched_features  : list = []

    for x in leave_out:
        if x in df.columns:
            untouched_features.append(x)

    for x in categorical:
        if x in df.columns:
            categorical_features.append(x)

        
    if metrics is None:
        metrics = [accuracy, BalancedAccuracy(), RocAuc(), MatthewsCorrCoef(), F1Score(average='macro'), Precision(average='macro'), Recall(average='macro')]


    continuous_features = list(set(df) - set(categorical_features) - set([target_label]) - set(untouched_features))

    splits = RandomSplitter(valid_pct=split, seed=seed)(range_of(df))
    

    # The dataframe is loaded into a fastai datastructure now that 
    # the feature engineering pipeline has been set up
    to = TabularPandas(
        df            , y_names=target_label                , 
        splits=splits , cat_names=categorical_features ,
        procs=procs   , cont_names=continuous_features , 
    )

    # The dataframe is then converted into a fastai dataset
    dls = to.dataloaders(bs=batch_size)

    # extract the file_name from the path
    p = pathlib.Path(file_name)
    file_name: str = str(p.parts[-1])


    learner = residual_tabular_learner(
        dls, 
        layers=list(shape), 
        metrics = metrics,
        cbs=callbacks,
        cardinality=cardinality
    )


    with learner.no_bar() if no_bar else contextlib.ExitStack() as gs:

        lr = learner.lr_find(suggest_funcs=[valley, slide, steep, minimum])

            # fitting functions, they give different results, some networks perform better with different learning schedule during fitting
        if(fit_choice == 'fit'):
            learner.fit(epochs, lr[lr_choice])
        elif(fit_choice == 'flat_cos'):
            learner.fit_flat_cos(epochs, lr[lr_choice])
        elif(fit_choice == 'one_cycle'):
            learner.fit_one_cycle(epochs, lr_max=lr[lr_choice])
        else:
            assert False, f'{fit_choice} is not a valid fit_choice'

        learner.recorder.plot_sched() 
        results = learner.validate()
        interp = ClassificationInterpretation.from_learner(learner)
        interp.plot_confusion_matrix()
                

    print(f'loss: {results[0]}, accuracy: {results[1]*100: .2f}%')
    learner.save(f'{file_name}.model')


    X_train = to.train.xs.reset_index(drop=True)
    X_test = to.valid.xs.reset_index(drop=True)
    y_train = to.train.ys.values.ravel()
    y_test = to.valid.ys.values.ravel()

    wrapped_model = SklearnWrapper(learner)

    classes = list(learner.dls.vocab)
    if len(classes) == 2:
        wrapped_model.target_type_ = 'binary'
    elif len(classes) > 2:  
        wrapped_model.target_type_ = 'multiclass'
    else:
        print('Must be more than one class to perform classification')
        raise ValueError('Wrong number of classes')
    
    wrapped_model._target_labels = target_label
    
    model_data: Model_data = Model_data(file_name, wrapped_model, classes, X_train, y_train, X_test, y_test, to, dls, name)


    return model_data



def run_tabnet_experiment(
    df: pd.DataFrame, 
    file_name: str, 
    target_label: str, 
    split=0.2, 
    name: str or None = None,
    categorical: list = ['Protocol'],
    procs = [FillMissing, Categorify, Normalize], 
    leave_out: list = [],
    epochs: int = 10,
    steps: int = 1,
    batch_size: int = 64,
    metrics: list or None = None,
    attention_size: int = 16,
    attention_width: int = 16,
    callbacks: list = [ShowGraphCallback],
    lr_choice: str = 'valley',
    fit_choice: str = 'flat_cos',
    no_bar: bool = False
) -> Model_data:
    '''
    Function trains a TabNet model on the dataframe and returns a model data named tuple
        Based on TabNet: Attentive Interpretable Tabular Learning by Sercan Arik and Tomas Pfister from Google Cloud AI (2016)
            where a DNN selects features from the input features based on an attention layer. Each step of the model selects 
            different features and uses the input from the previous step to ultimately make predictions
    
        Combines aspects of a transformer, decision trees, and deep neural networks to learn tabular data, and has achieved state
            of the art results on some datasets.

        Capable of self-supervised learning, however it is not implemented here yet.

    (https://arxiv.org/pdf/1908.07442.pdf)

    Parameters:
        df: pandas dataframe containing the data
        file_name: name of the file the dataset came from
        target_label: the label to predict
        name: name of the experiment, if none a default is given
        split: the percentage of the data to use for testing
        categorical: list of the categorical columns
        procs: list of preprocessing functions to apply in the dataloaders pipeline
                additional options are: 
                    PCA_tabular (generate n principal components) 
                    Normal (features are scaled to the interval [0,1])
        leave_out: list of columns to leave out of the experiment
        epochs: number of epochs to train for  
        batch_size: number of samples processed in one forward and backward pass of the model
        metrics: list of metrics to calculate and display during training
        attention size: determines the number of rows and columns in the attention layers
        attention width: determines the width of the decision layer
        callbacks: list of callbacks to apply during training
        lr_choice: where the learning rate sampling function should find the optimal learning rate
                    choices are: 'valley', 'steep', 'slide', and 'minimum'
        fit_choice: choice of function that controls the learning schedule choices are:
                    'fit': a standard learning schedule 
                    'flat_cos': a learning schedule that starts high before annealing to a low value
                    'one_cyle': a learning schedule that warms up for the first epochs, continues at a high
                                    learning rate, and then cools down for the last epochs
                    the default is 'flat_cos'

    
    returns a model data named tuple
        model_data: tuple = (file_name, model, classes, X_train, y_train, X_test, y_test, model_type)
    '''

    if name is None:
        name = f"TabNet_steps_{steps}_width_{attention_width}_attention_{attention_size}"

    lr_choice = {'valley': 0, 'slide': 1, 'steep': 2, 'minimum': 3}[lr_choice]


    categorical_features: list = []
    untouched_features  : list = []

    for x in leave_out:
        if x in df.columns:
            untouched_features.append(x)

    for x in categorical:
        if x in df.columns:
            categorical_features.append(x)

        
    if metrics is None:
        metrics = [accuracy, BalancedAccuracy(), RocAuc(), MatthewsCorrCoef(), F1Score(average='macro'), Precision(average='macro'), Recall(average='macro')]


    continuous_features = list(set(df) - set(categorical_features) - set([target_label]) - set(untouched_features))

    splits = RandomSplitter(valid_pct=split, seed=seed)(range_of(df))
    
    # The dataframe is loaded into a fastai datastructure now that 
    # the feature engineering pipeline has been set up

    to = TabularPandas(
        df            , y_names=target_label                , 
        splits=splits , cat_names=categorical_features ,
        procs=procs   , cont_names=continuous_features , 
    )

    # The dataframe is then converted into a fastai dataset
    dls = to.dataloaders(bs=batch_size)

    # extract the file_name from the path
    p = pathlib.Path(file_name)
    file_name: str = str(p.parts[-1])


    emb_szs = get_emb_sz(to)


    net = TabNet(emb_szs, len(to.cont_names), dls.c, n_d=attention_width, n_a=attention_size, n_steps=steps) 
    tab_model = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=metrics, opt_func=ranger, cbs=callbacks)


    with tab_model.no_bar() if tab_model else contextlib.ExitStack() as gs:

        lr = tab_model.lr_find(suggest_funcs=[valley, slide, steep, minimum])


        # fitting functions, they give different results, some networks perform better with different learning schedule during fitting
        if(fit_choice == 'fit'):
            tab_model.fit(epochs, lr[lr_choice])
        elif(fit_choice == 'flat_cos'):
            tab_model.fit_flat_cos(epochs, lr[lr_choice])
        elif(fit_choice == 'one_cycle'):
            tab_model.fit_one_cycle(epochs, lr_max=lr[lr_choice])
        else:
            assert False, f'{fit_choice} is not a valid fit_choice'


        tab_model.recorder.plot_sched() 
        results = tab_model.validate()
        interp = ClassificationInterpretation.from_learner(tab_model)
        interp.plot_confusion_matrix()
                

    tab_model.save(f'{file_name}.model')
    print(f'loss: {results[0]}, accuracy: {results[1]*100: .2f}%')


    X_train = to.train.xs.reset_index(drop=True)
    X_test = to.valid.xs.reset_index(drop=True)
    y_train = to.train.ys.values.ravel()
    y_test = to.valid.ys.values.ravel()

    wrapped_model = SklearnWrapper(tab_model)

    classes = list(tab_model.dls.vocab)
    if len(classes) == 2:
        wrapped_model.target_type_ = 'binary'
    elif len(classes) > 2:  
        wrapped_model.target_type_ = 'multiclass'
    else:
        print('Must be more than one class to perform classification')
        raise ValueError('Wrong number of classes')
    
    wrapped_model._target_labels = target_label
    
    model_data: Model_data = Model_data(file_name, wrapped_model, classes, X_train, y_train, X_test, y_test, to, dls, name)


    return model_data


def decode_classes_and_create_Xy_df(model_data: Model_data, target_label: str):
    """
        Function takes a Model_data namedtuple and returns a dataframe with the X and decoded y data
    """
    X = model_data.X_train
    y = model_data.y_train
    classes = model_data.classes

    Xy_df = pd.DataFrame(X)
    y_s: list = []
    for x in y:
        y_s.append(classes[x])
    Xy_df[target_label] = y_s

    return Xy_df


def normalize_bilabel_dataset_between_0_1(df: pd.DataFrame, labels = ['Traffic Type', 'Application Type']) -> pd.DataFrame:
    '''
        Function takes a dataframe and merges its labels before normalizing data. 
        The labels are then split back into their original form, but the merged label is kept for verification purposes.

        returns a trilabel dataframe with the new label under the 'Merged Type' column
    '''
    inter_df = df.copy()

    d1_y = inter_df[labels[0]]
    d2_y = inter_df[labels[1]]

    merged_y = pd.concat([d1_y, d2_y], axis=1)

    merged: list = []
    for x in zip(d1_y, d2_y):
        merged.append(f'{x[0]}_{x[1]}')

    inter_df['Merged Type'] = merged

    normalized_model_data = transform_and_split_data(
        inter_df,
        'Merged Type',
        split=0,
        scale=True,
        categorical=[],
        leave_out=labels
    )

    merged_df = decode_classes_and_create_Xy_df(normalized_model_data, 'Merged Type')

    merged_y = merged_df['Merged Type']
    new_labels: list = []

    for l in labels:
        new_labels.append([])

    for x in merged_y:
        split_label = x.split('_')

        for i in range(len(split_label)):
            new_labels[i].append(split_label[i])

    for i, x in enumerate(labels):
        merged_df[x] = new_labels[i]

    total_df = merged_df.copy()
    total_df['Merged Type'] = merged_y


    return total_df

    

def create_n_cloud_from_dataset(df: pd.DataFrame, n_points: int = 100, target_label: str = 'label', allow_partial: bool = False, segmented=True):
    '''
        Function takes a dataframe and splits it into point clouds with n_points each. 
            The generated clouds will all have the indicated number of points, discarding the remainder if allow_partial is False.
            The generated clouds will be partitioned by the classes contained within the target_label column.
            Each cloud will be labeled with the class it was generated from.

            If the segmented parameter is true, the clouds will be split into df.shape[0]/n_points number of clouds.
                This is because we segment the dataframe to produce the clouds.
            if the segmented parameter is false, the clouds will be split into df.shape[0] - n_points number of clouds.
                This is because we will be using a sliding interval to construct the clouds.

        returns a numpy array of point clouds as well as a numpy array with class labels and class mappings.
    '''

    data = transform_and_split_data(df, target_label=target_label, split=0)
    X = data.X_train.copy()
    X[target_label] = data.y_train.copy()

    classes = data.classes

    # create n clouds by class
    clouds          : list = []
    clouds_y        : list = []
    clouds_y_decoded: list = []

    for i, x in enumerate(classes):
        temp_df = X[X[target_label] == i]
        if(segmented):
            iterations: int = int(temp_df.shape[0]/n_points)
        else:
            iterations: int = temp_df.shape[0] - n_points


        for j in range(int(iterations)):
            if(segmented):
                cloud = temp_df[j*n_points:(j+1)*n_points].drop(target_label, axis=1).values
            else:
                cloud = temp_df[j:j+n_points].drop(target_label, axis=1).values
            
            keep: bool = allow_partial and not segmented
            if(len(cloud) == n_points or keep):
                clouds.append(cloud)
                clouds_y.append(i)
                clouds_y_decoded.append(x)

    clouds = np.array(clouds)
    clouds_y = np.array(clouds_y)
    clouds_y_decoded = np.array(clouds_y_decoded)

    return clouds, clouds_y, classes, clouds_y_decoded


def calculate_correlations(model_data: Model_data, target_label: str):
    '''
        Function merges together the encoded and standardized model data and labels to calculate pearson correlation
    '''

    encoded_data = model_data.X_train.copy()
    encoded_data[target_label] = model_data.y_train

    return encoded_data.corr()


def extract_correlations(correlations: pd.DataFrame, feature: str, leave_out: list = ['Traffic Type', 'Application Type']) -> list:
    '''
        Function takes a correlation dataframe and extracts a list of features correlated with the input feature. 
            Anything in leave_out is not included in the list.
    '''

    correlation_order: list = list(correlations.sort_values(by=feature, ascending=False).index)

    for x in leave_out:
        if x in correlation_order:
            correlation_order.remove(x)

    return correlation_order


def visualize_confusion_matrix(model_data: tuple, ax=None, cmap='Blues') -> yb.classifier:
    '''
        Takes a 10-tuple from the run_experiments function and creates a confusion matrix

        model_data: tuple = (name, model, classes, X_train, y_train, X_test, y_test, to, dls, model_type)
    '''

    visualizer = yb.classifier.ConfusionMatrix(model_data[1], classes=model_data[2], title=model_data[0], ax=ax, cmap=cmap, is_fitted=True)
    visualizer.score(model_data[5], model_data[6])
    
    return visualizer


def visualize_roc(model_data: tuple, ax=None, cmap='Blues') -> yb.classifier:
    '''
        Takes a 10-tuple from the run_experiments function and creates a 
        Receiver Operating Characteristic (ROC) Curve

        model_data: tuple = (name, model, classes, X_train, y_train, X_test, y_test, to, dls, model_type)
    '''

    visualizer = yb.classifier.ROCAUC(model_data[1], classes=model_data[2], title=model_data[0], ax=ax, cmap=cmap)
    visualizer.score(model_data[5], model_data[6])
    
    return visualizer



def visualize_pr_curve(model_data: tuple, ax=None, cmap='Blues') -> yb.classifier:
    '''
        Takes a 10-tuple from the run_experiments function and creates a 
        Precision-Recall Curve

        model_data: tuple = (name, model, classes, X_train, y_train, X_test, y_test, to, dls, model_type)
    '''

    visualizer = yb.classifier.PrecisionRecallCurve(model_data[1], title=model_data[0], ax=ax, cmap=cmap)
    visualizer.score(model_data[5], model_data[6])

    return visualizer



def visualize_report(model_data: tuple, ax=None, cmap='Blues') -> yb.classifier:
    '''
        Takes a 10-tuple from the run_experiments function and creates a report
        detailing the Precision, Recall, f1, and Support scores for all 
        classification outcomes

        model_data: tuple = (name, model, classes, X_train, y_train, X_test, y_test, to, dls, model_type)
    '''

    visualizer = yb.classifier.ClassificationReport(model_data[1], classes=model_data[2], title=model_data[0], support=True, ax=ax, cmap=cmap)
    visualizer.score(model_data[5], model_data[6])

    return visualizer



def visualize_class_balance(model_data: tuple, ax=None, cmap='Blues') -> yb.classifier:
    '''
        Takes a 10-tuple from the run_experiments function and creates a histogram
        detailing the balance between classification outcomes

        model_data: tuple = (name, model, classes, X_train, y_train, X_test, y_test, to, dls, model_type)
    '''

    visualizer = yb.target.ClassBalance(labels=model_data[0], ax=ax, cmap=cmap)
    visualizer.fit(model_data[4], model_data[6])
    
    return visualizer



def confusion_matrix_from_dataset(model_data: tuple, df: pd.DataFrame, ax=None, cmap='Blues') -> yb.classifier:
    '''
        Takes a 10-tuple from the run_experiments function and uses the model to classify
            the data passed in as the dataframe. Produces a confusion matrix
            (currently only confirmed to work with fastai models)

        model_data: tuple = (name, model, classes, X_train, y_train, X_test, y_test, to, dls, model_type)
    '''

    dl = model_data[1].model.dls.test_dl(df, bs=64)
    preds, v, dec_preds = model_data[1].model.get_preds(dl=dl, with_decoded=True)

    visualizer = yb.classifier.ConfusionMatrix(model_data[1], classes=model_data[2], title=model_data[0], ax=ax, cmap=cmap)
    visualizer.score(dl.xs, dl.y)
    acc = accuracy_score(dl.y, dec_preds)
    print(f'Accuracy: {acc}')

    return visualizer


"""def show_stacked_graphs(figures: list, stack_shape: tuple = (1,1), title: str = '', individual_titles: list or str = None) -> widgets.VBox: 
    
    Function takes a list of figures and stacks them in a grid.
        stack_shape is a tuple of the number of rows and columns in the grid and
        the product of the indices must equal the number of figures.
        individual_titles allows one to specify titles for each figure, or a single title for all figures.

    Returns a plotly figure with the input figures stacked as indicated
    
    if(individual_titles is not None and type(individual_titles) is not str):
        assert len(figures) == len(individual_titles)     , "Number of titles must match number of figures"
    assert len(figures) == stack_shape[0] * stack_shape[1], "Number of figures must match stack shape"

    vertical_stack  : list = []
    horizantal_stack: list = []

    if(title != ''):
        vertical_stack.append(widgets.Label(title))
    
    for i in range(stack_shape[1]):
        for j in range(stack_shape[0]):
                
            temp_fig = go.FigureWidget(figures[i * stack_shape[0] + j].data)
            if(type(individual_titles) == str):
                temp_fig.layout.title = individual_titles
            elif(type(individual_titles) == list):
                temp_fig.layout.title = individual_titles[i * stack_shape[0] + j]
                
            horizantal_stack.append(temp_fig)
        vertical_stack.append(widgets.HBox(horizantal_stack))
        horizantal_stack = []

    fig = widgets.VBox(vertical_stack)

    return fig """


def show_clusters_by_label_2D(model_data: Model_data, x_col: str, y_col: str, target_label: str = 'label', samples=25000, title='Clusters by Label', ) -> dict:
    '''
        Takes a model_data tuple and displays the indicated columns of the X_train data as a scatter plot.
    '''

    X            = model_data.X_train.copy()
    X[target_label]   = model_data.y_train.copy()
    classes      = model_data.classes
    num_labels   = len(X.groupby(target_label).size())
    label_colors = get_n_color_list(num_labels, opacity=.4)



    # _X            = pd.concat([X, component], axis=1, join='inner')
    new_X         = pd.DataFrame(np.array(X.sample(samples)))
    new_X.columns = X.columns


    traces: list = []
    for i in range(num_labels):
        cluster = new_X[new_X[target_label] == i]
        traces.append(
            go.Scatter(
                x      = cluster[x_col]            ,
                y      = cluster[y_col]            ,
                marker = {'color': label_colors[i]},
                mode   = 'markers'                 ,
                name   = classes[i]                ,
            )
        )


    layout = dict(
        title = title,
        xaxis = dict(
            title    = x_col,
            ticklen  = 5    ,
            zeroline = False,
        ),
        yaxis = dict(
            title    = y_col,
            ticklen  = 5    ,
            zeroline = False,
        )
    )

    fig: dict = {'data': traces, 'layout': layout}
    fig = go.Figure(fig)

    return fig


def faceted_clusters_by_label_2D(model_data: Model_data, x_col: str, y_col: str, target_label: str = 'label', samples=25000, title='Clusters by Label', ) -> dict:
    '''
        Takes a model_data tuple and returns a list of figures that each contain a highlighted value in the label with the other values
            serving as a grey background in a scatter plot that demonstrates the distribution
    '''

    X            = model_data.X_train.copy()
    X[target_label]   = model_data.y_train.copy()
    classes      = model_data.classes
    num_labels   = len(X.groupby(target_label).size()) + 1
    label_colors = get_n_color_list(num_labels, opacity=.7)



    new_X         = pd.DataFrame(np.array(X.sample(samples)))
    new_X.columns = X.columns

    figs: list = []
    for i, x in enumerate(classes):

        cluster_1 = new_X[new_X[target_label] == i]
        cluster_2 = new_X[new_X[target_label] != i]

        traces: list = []

        traces.append(
            go.Scatter(
                x      = cluster_2[x_col]                     ,
                y      = cluster_2[y_col]                     ,
                marker = {'color': 'rgba(170, 170, 170, 0.5)'},
                mode   = 'markers'                            ,
                name   = 'Other'                              ,
            )
        )

        traces.append(
            go.Scatter(
                x      = cluster_1[x_col]          ,
                y      = cluster_1[y_col]          ,
                marker = {'color': label_colors[i]},
                mode   = 'markers'                 ,
                name   = x                         ,
            )
        )



        layout = dict(
            title = title,
            xaxis = dict(
                title    = x_col,
                ticklen  = 5    ,
                zeroline = False,
            ),
            yaxis = dict(
                title    = y_col,
                ticklen  = 5    ,
                zeroline = False,
            )
        )

        fig = go.Figure({'data': traces, 'layout': layout})
        # a, b = fig.data
        # fig.data = b, a
        figs.append(fig)

    return figs


def visualize_side_by_side(
    model_datum: list,
    results: dict,
    title: str = "Confusion Matrices",
    model_descriptions: list or None = None,
    plotting_function: callable = visualize_confusion_matrix,
    shape: tuple = (2,5),
    size: tuple = (20,10),
    x_label: str = 'Predicted',
    y_label: str = 'True',
) -> tuple:
    '''
        Function will take the plotting function and execute it on each Model_data tuple passed in through the model_datum list
            The plots will be oriented in a subplot grid with the number of rows and columns specified by the shape tuple
            average accuracy will be calculated and displayed in the subtitle of the figure
            
    '''

    print('Ignore yellowbrick warnings, this is a side-effect of using the sklearn wrapper on the fastai model')
    rows = shape[0]
    cols = shape[1]

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=size)
    fig.suptitle(title)
    fig.text(0.5, .001, f'    Average Accuracy: {np.average(results["accuracy"]) * 100: .4f}', ha='center')

    viz: list = [0] * len(model_datum)
    for i in range(rows*cols):
        row = i // cols
        col = i % cols
        if i < len(model_datum):
            if(rows == 1):
                current_ax = ax[col]
            else:
                current_ax = ax[row][col]
            viz[i] = plotting_function(model_datum[i], ax=current_ax)
            viz[i].finalize()
            if model_descriptions is not None:
                current_ax.set_title(model_descriptions[i])

        if(row == rows-1):
            current_ax.set_xlabel(x_label)
        else:
            current_ax.set_xlabel('')
            current_ax.xaxis.set_ticklabels([])

        if(col == 0):
            current_ax.set_ylabel(y_label)
        else:
            current_ax.set_ylabel('')
            current_ax.yaxis.set_ticklabels([])

    plt.tight_layout()

    return (fig, ax)


def show_feature_importance_TA(model: Learner, df: pd.DataFrame, ax = None, figsize = (15,5), title='Feature Importances') -> tuple:
    '''
        Function shows the feature importance calculated by the attention layer of the TabNet model.

    '''

    dl = model.dls.test_dl(df, bs=64)
    feature_importances = tabnet_feature_importances(model.model, dl)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.bar(dl.x_names, feature_importances, color='g')
    ax.set_xticks(range(len(feature_importances)))
    ax.set_xticklabels(dl.x_names, rotation=90)
    ax.set_title(title)

    return (fig, ax)


def plot_point_clouds(clouds: list, shape: tuple = (1, 4), titles: list or None = None, homology_dimensions: list or None = None, color='magma'):


    specs = np.zeros(shape).astype(object)
    cols = shape[1]

    for i in range(specs.shape[0]):
        for j in range(specs.shape[1]):
            specs[i, j] = {'type': 'scene'}


    fig = make_subplots(rows=shape[0], cols=shape[1], specs=specs.tolist(), column_titles=titles)

    for i, cloud in enumerate(clouds):
        row = (i // cols) + 1
        col = (i % cols) + 1

        fig.add_trace(go.Scatter3d(
            # x=cloud[0, :],
            # y=cloud[1, :],
            # z=cloud[2, :],
            x=cloud[:, 0],
            y=cloud[:, 1],
            z=cloud[:, 2],
            showlegend=False,
            mode="markers",
            marker={"size": 4,
                    "color": list(range(cloud.shape[0])),
                    "colorscale": color,
                    "opacity": 0.8}
            ),
            row=row, col=col
        )

    fig.show()

    return fig


def plot_persistence_diagrams(diagrams: list, shape: tuple = (1, 4), titles: list or None = None, homology_dimensions: list or None = None, colors: list or None = None):
    '''
        Function plots a set of persistence diagrams
    '''


    cols = shape[1]
    fig = make_subplots(rows=shape[0], cols=cols, column_titles=titles)

    if homology_dimensions is None:
        homology_dimensions = np.unique(diagrams[0][:, 2])

    if colors is None:
        colors = get_n_color_list(len(homology_dimensions), opacity=.8)

    for i, diagram in enumerate(diagrams):
        row = (i // cols) + 1
        col = (i % cols) + 1


        diagram = diagram[diagram[:, 0] != diagram[:, 1]]
        diagram_no_dims = diagram[:, :2]
        posinfinite_mask = np.isposinf(diagram_no_dims)
        neginfinite_mask = np.isneginf(diagram_no_dims)
        max_val = np.max(np.where(posinfinite_mask, -np.inf, diagram_no_dims))
        min_val = np.min(np.where(neginfinite_mask, np.inf, diagram_no_dims))
        parameter_range = max_val - min_val
        extra_space_factor = 0.02
        has_posinfinite_death = np.any(posinfinite_mask[:, 1])

        if has_posinfinite_death:
            posinfinity_val = max_val + 0.1 * parameter_range
            extra_space_factor += 0.1

        extra_space = extra_space_factor * parameter_range
        min_val_display = min_val - extra_space
        max_val_display = max_val + extra_space

        fig.add_trace(
            go.Scatter(
                x = [min_val_display, max_val_display],
                y = [min_val_display, max_val_display],
                mode       = "lines",
                line       = {"dash": "dash", "width": 1, "color": "black"},
                showlegend = False,
                hoverinfo  = "none"
            ), 
            row = row, 
            col = col
        )

        for k, dim in enumerate(homology_dimensions):
            name = f"{int(dim)}d Persistence" if dim != np.inf else "Persistence"
            subdiagram = diagram[diagram[:, 2] == dim]
            unique, inverse, counts = np.unique(subdiagram, axis=0, return_inverse=True, return_counts=True)
            
            hovertext = [
                f"{tuple(unique[unique_row_index][:2])}" + (
                    f", multiplicity: {counts[unique_row_index]}" if counts[unique_row_index] > 1 else ""
                ) for unique_row_index in inverse
            ]

            y = subdiagram[:, 1]
            if has_posinfinite_death:
                y[np.isposinf(y)] = posinfinity_val

            fig.add_trace(
                go.Scatter(
                    x = subdiagram[:, 0], 
                    y = y, 
                    hoverinfo  = "text", hovertext=hovertext, name=name,
                    marker     = {'color': colors[k]},
                    mode       = "markers", 
                    showlegend = (i==0),
                ), 
                row=row, 
                col=col
            )


        if(row == (len(diagrams)//cols)):
            fig.update_xaxes(title_text="Feature Birth", row=row, col=col)
        
        if(col == 1):
            fig.update_yaxes(title_text="Feature Death", row=row, col=col)


        if has_posinfinite_death:
            fig.add_trace(
                go.Scatter(
                    x = [min_val_display, max_val_display],
                    y = [posinfinity_val, posinfinity_val],
                    line = {"dash": "dash", "width": 0.5, "color": "black"},
                    hoverinfo = "none",
                    showlegend = True,
                    name = u"\u221E",
                    mode = "lines",
                ), 
                row=row, 
                col=col
            )

    fig.show()

    return fig


def plot_point_clouds_and_persistence_diagram(point_clouds: list, persistence_diagrams: list, rows: int, columns: int, homology_dimensions: list or None = None, color='magma', column_titles: list or None = None):



    cols = columns

    ri1 = {'type': 'scene'}
    ri2 = {'type': 'xy'}
    specs = np.zeros((2,4)).astype(object)

    for i in range(specs.shape[0]):
        for j in range(specs.shape[1]):
            if(i == 0):
                specs[i, j] = ri1
            else:
                specs[i, j] = ri2


    if homology_dimensions is None:
        homology_dimensions = np.unique(persistence_diagrams[0][:, 2])

    label_colors = get_n_color_list(len(homology_dimensions), opacity=.8)


    fig = make_subplots(rows=rows, cols=cols, specs=specs.tolist(), column_titles=column_titles)



    for i, diagram in enumerate(persistence_diagrams):
        row = (i // cols) + 1
        col = (i % cols) + 1

        if(i >= rows*cols):
            break

        fig.add_trace(go.Scatter3d(
            x=point_clouds[i][0, :],
            y=point_clouds[i][1, :],
            z=point_clouds[i][2, :],
            # x=point_clouds[i][:, 0],
            # y=point_clouds[i][:, 1],
            # z=point_clouds[i][:, 2],
            showlegend=False,
            mode="markers",
            marker={"size": 4,
                    "color": list(range(point_clouds[j].shape[0])),
                    "colorscale": color,
                    "opacity": 0.8}
            ),
            row=row, col=col
        )


        row += 1

        diagram = diagram[diagram[:, 0] != diagram[:, 1]]
        diagram_num_dims = diagram[:, :2]
        posinfinite_mask = np.isposinf(diagram_num_dims)
        neginfinite_mask = np.isneginf(diagram_num_dims)
        max_val = np.max(np.where(posinfinite_mask, -np.inf, diagram_num_dims))
        min_val = np.min(np.where(neginfinite_mask, np.inf, diagram_num_dims))
        parameter_range = max_val - min_val
        extra_space_factor = 0.02
        has_posinfinite_death = np.any(posinfinite_mask[:, 1])

        if has_posinfinite_death:
            posinfinity_val     = max_val + 0.1 * parameter_range
            extra_space_factor += 0.1


        extra_space     = extra_space_factor * parameter_range
        min_val_display = min_val - extra_space
        max_val_display = max_val + extra_space

        fig.add_trace(
            go.Scatter(
                x = [min_val_display, max_val_display],
                y = [min_val_display, max_val_display],
                mode = "lines",
                line = {"dash": "dash", "width": 1, "color": "black"},
                showlegend = False,
                hoverinfo = "none"
            ), 
            row = row, 
            col = col
        )

        for k, dim in enumerate(homology_dimensions):

            name = f"{int(dim)}d Persistence" if dim != np.inf else "Persistence"
            subdiagram = diagram[diagram[:, 2] == dim]
            unique, inverse, counts = np.unique(subdiagram, axis=0, return_inverse=True, return_counts=True)
            
            hovertext = [
                f"{tuple(unique[unique_row_index][:2])}" + (
                    f", multiplicity: {counts[unique_row_index]}"
                    if counts[unique_row_index] > 1 else ""
                ) for unique_row_index in inverse
            ]

            y = subdiagram[:, 1]
            if has_posinfinite_death:
                y[np.isposinf(y)] = posinfinity_val

            fig.add_trace(
                go.Scatter(
                    x = subdiagram[:, 0], 
                    y = y, 
                    name       = name     , 
                    mode       = "markers", 
                    hoverinfo  = "text"   , 
                    hovertext  = hovertext, 
                    showlegend = (i==0)   ,
                    marker     = {'color': label_colors[k]},
                ), 
                row = row, 
                col = col
            )


        if(row == 2):
            fig.update_xaxes(title_text="Feature Birth", row=row, col=col)
        
        if(col == 1):
            fig.update_yaxes(title_text="Feature Death", row=row, col=col)

        # Add a horizontal dashed line for points with infinite death
        if has_posinfinite_death:
            fig.add_trace(go.Scatter(
                x=[min_val_display, max_val_display],
                y=[posinfinity_val, posinfinity_val],
                mode="lines",
                line={"dash": "dash", "width": 0.5, "color": "black"},
                showlegend=True,
                name=u"\u221E",
                hoverinfo="none"
            ), row=row, col=col)

    fig.show()

    return fig
