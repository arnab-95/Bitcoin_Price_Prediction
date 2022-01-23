# %% [code] {"papermill":{"duration":2.139532,"end_time":"2021-03-06T09:02:31.209186","exception":false,"start_time":"2021-03-06T09:02:29.069654","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:18.540166Z","iopub.execute_input":"2021-11-28T01:10:18.540504Z","iopub.status.idle":"2021-11-28T01:10:24.169348Z","shell.execute_reply.started":"2021-11-28T01:10:18.540466Z","shell.execute_reply":"2021-11-28T01:10:24.168343Z"}}

import numpy as np
import pandas as pd
from datetime import datetime

import seaborn as sns

sns.set(rc={'figure.figsize': (10, 6)})
custom_colors = ["#4e89ae", "#c56183", "#ed6663", "#ffa372"]

import matplotlib.pyplot as plt
% matplotlib
inline
import matplotlib.image as mpimg

# Colorama
from colorama import Fore, Back, Style

y_ = Fore.CYAN
m_ = Fore.WHITE

import networkx as nx
import plotly.graph_objects as go

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

# %% [code] {"papermill":{"duration":0.050253,"end_time":"2021-03-06T09:02:31.362584","exception":false,"start_time":"2021-03-06T09:02:31.312331","status":"completed"},"tags":[],"scrolled":true,"execution":{"iopub.status.busy":"2021-11-28T01:10:24.171341Z","iopub.execute_input":"2021-11-28T01:10:24.171705Z","iopub.status.idle":"2021-11-28T01:10:24.179134Z","shell.execute_reply.started":"2021-11-28T01:10:24.171636Z","shell.execute_reply":"2021-11-28T01:10:24.178264Z"}}
print(f"{m_}Total no. of records:{y_}{df.shape}\n")
print(f"{m_}Corresponding Data types of data columns: \n{y_}{df.dtypes}")

# %% [code] {"execution":{"iopub.status.busy":"2021-11-28T01:10:24.181019Z","iopub.execute_input":"2021-11-28T01:10:24.18134Z","iopub.status.idle":"2021-11-28T01:10:32.207635Z","shell.execute_reply.started":"2021-11-28T01:10:24.181297Z","shell.execute_reply":"2021-11-28T01:10:32.20674Z"}}
df['Timestamp'] = [datetime.fromtimestamp(i) for i in df['Timestamp']]
df = df.set_index('Timestamp')
df = df.resample("24H").mean()
df.head()

# %% [code] {"papermill":{"duration":0.286599,"end_time":"2021-03-06T09:02:32.536099","exception":false,"start_time":"2021-03-06T09:02:32.2495","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:32.210123Z","iopub.execute_input":"2021-11-28T01:10:32.21036Z","iopub.status.idle":"2021-11-28T01:10:32.528875Z","shell.execute_reply.started":"2021-11-28T01:10:32.210332Z","shell.execute_reply":"2021-11-28T01:10:32.527851Z"}}
values_missed = pd.DataFrame()
values_missed['column'] = df.columns

values_missed['percent'] = [round(100 * df[z].isnull().sum() / len(df), 2) for z in df.columns]
values_missed = values_missed.sort_values('percent', ascending=False)
values_missed = values_missed[values_missed['percent'] > 0]

fig = sns.barplot(
    x=values_missed['percent'],
    y=values_missed["column"],
    orientation='horizontal', palette="winter"
).set_title('Missed values percent for every column')


# %% [code] {"papermill":{"duration":0.054573,"end_time":"2021-03-06T09:02:32.707766","exception":false,"start_time":"2021-03-06T09:02:32.653193","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:32.530455Z","iopub.execute_input":"2021-11-28T01:10:32.530725Z","iopub.status.idle":"2021-11-28T01:10:32.546328Z","shell.execute_reply.started":"2021-11-28T01:10:32.530687Z","shell.execute_reply":"2021-11-28T01:10:32.545331Z"}}
def filling_the_misssing_values(df):
    df['Open'] = df['Open'].interpolate()
    df['Close'] = df['Close'].interpolate()
    df['Weighted_Price'] = df['Weighted_Price'].interpolate()
    df['Volume_(BTC)'] = df['Volume_(BTC)'].interpolate()
    df['Volume_(Currency)'] = df['Volume_(Currency)'].interpolate()
    df['High'] = df['High'].interpolate()
    df['Low'] = df['Low'].interpolate()
    print(f'{m_}After interploation,No. of Missing values:\n{y_}{df.isnull().sum()}')


filling_the_misssing_values(df)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-28T01:10:32.54797Z","iopub.execute_input":"2021-11-28T01:10:32.548307Z","iopub.status.idle":"2021-11-28T01:10:32.554282Z","shell.execute_reply.started":"2021-11-28T01:10:32.548251Z","shell.execute_reply":"2021-11-28T01:10:32.55368Z"}}
df.columns

# %% [code] {"execution":{"iopub.status.busy":"2021-11-28T01:10:32.555439Z","iopub.execute_input":"2021-11-28T01:10:32.556246Z","iopub.status.idle":"2021-11-28T01:10:32.578745Z","shell.execute_reply.started":"2021-11-28T01:10:32.55621Z","shell.execute_reply":"2021-11-28T01:10:32.577714Z"}}
naya_df = df.groupby('Timestamp').mean()
naya_df = naya_df[['Volume_(BTC)', 'Close', 'Volume_(Currency)']]
naya_df.rename(
    columns={'Volume_(BTC)': 'Volume_market_mean', 'Close': 'close_mean', 'Volume_(Currency)': 'volume_curr_mean'},
    inplace=True)
naya_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-28T01:10:32.582286Z","iopub.execute_input":"2021-11-28T01:10:32.583249Z","iopub.status.idle":"2021-11-28T01:10:32.615695Z","shell.execute_reply.started":"2021-11-28T01:10:32.583214Z","shell.execute_reply":"2021-11-28T01:10:32.614827Z"}}
merged_df = df.merge(naya_df, left_on='Timestamp',
                     right_index=True)
merged_df['volume(BTC)/Volume_market_mean'] = merged_df['Volume_(BTC)'] / merged_df['Volume_market_mean']
merged_df['Volume_(Currency)/volume_curr_mean'] = merged_df['Volume_(Currency)'] / merged_df['volume_curr_mean']

merged_df['close/close_market_mean'] = merged_df['Close'] / merged_df['close_mean']
merged_df['open/close'] = merged_df['Open'] / merged_df['Close']
merged_df["gap"] = merged_df["High"] - merged_df["Low"]
merged_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-28T01:10:32.616901Z","iopub.execute_input":"2021-11-28T01:10:32.617217Z","iopub.status.idle":"2021-11-28T01:10:32.631951Z","shell.execute_reply.started":"2021-11-28T01:10:32.617183Z","shell.execute_reply":"2021-11-28T01:10:32.631234Z"}}

merged_df.info()


# %% [code] {"papermill":{"duration":0.051824,"end_time":"2021-03-06T09:02:32.956506","exception":false,"start_time":"2021-03-06T09:02:32.904682","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:32.63457Z","iopub.execute_input":"2021-11-28T01:10:32.635272Z","iopub.status.idle":"2021-11-28T01:10:32.643111Z","shell.execute_reply.started":"2021-11-28T01:10:32.635203Z","shell.execute_reply":"2021-11-28T01:10:32.642486Z"}}
def tigna_plot(x, title, c):
    fig, ax = plt.subplots(3, 1, figsize=(25, 10), sharex=True)
    sns.distplot(x, ax=ax[0], color=c)
    ax[0].set(xlabel=None)
    ax[0].set_title('Histogram + KDE')
    sns.boxplot(x, ax=ax[1], color=c)
    ax[1].set(xlabel=None)
    ax[1].set_title('Boxplot')
    sns.violinplot(x, ax=ax[2], color=c)
    ax[2].set(xlabel=None)
    ax[2].set_title('Violin plot')
    fig.suptitle(title, fontsize=30)
    plt.tight_layout(pad=3.0)
    plt.show()


# %% [code] {"papermill":{"duration":0.685832,"end_time":"2021-03-06T09:02:33.680222","exception":false,"start_time":"2021-03-06T09:02:32.99439","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:32.644303Z","iopub.execute_input":"2021-11-28T01:10:32.64489Z","iopub.status.idle":"2021-11-28T01:10:33.622808Z","shell.execute_reply.started":"2021-11-28T01:10:32.644857Z","shell.execute_reply":"2021-11-28T01:10:33.621825Z"}}
tigna_plot(df['Open'], 'Distribution of Opening price', custom_colors[0])

# %% [code] {"papermill":{"duration":0.656125,"end_time":"2021-03-06T09:02:34.378142","exception":false,"start_time":"2021-03-06T09:02:33.722017","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:33.624382Z","iopub.execute_input":"2021-11-28T01:10:33.624744Z","iopub.status.idle":"2021-11-28T01:10:34.394455Z","shell.execute_reply.started":"2021-11-28T01:10:33.624698Z","shell.execute_reply":"2021-11-28T01:10:34.393612Z"}}
tigna_plot(df['High'], 'Distribution of the highest price', custom_colors[1])

# %% [code] {"papermill":{"duration":0.794204,"end_time":"2021-03-06T09:02:35.217751","exception":false,"start_time":"2021-03-06T09:02:34.423547","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:34.396106Z","iopub.execute_input":"2021-11-28T01:10:34.396362Z","iopub.status.idle":"2021-11-28T01:10:35.161783Z","shell.execute_reply.started":"2021-11-28T01:10:34.396332Z","shell.execute_reply":"2021-11-28T01:10:35.161172Z"}}
tigna_plot(df['Low'], 'Distribution of Lowest Price', custom_colors[2])

# %% [code] {"papermill":{"duration":0.67007,"end_time":"2021-03-06T09:02:35.931821","exception":false,"start_time":"2021-03-06T09:02:35.261751","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:35.162702Z","iopub.execute_input":"2021-11-28T01:10:35.163414Z","iopub.status.idle":"2021-11-28T01:10:35.924493Z","shell.execute_reply.started":"2021-11-28T01:10:35.163376Z","shell.execute_reply":"2021-11-28T01:10:35.923609Z"}}
tigna_plot(df['Close'], 'Distribution of the closing Price', custom_colors[3])

# %% [code] {"papermill":{"duration":0.685952,"end_time":"2021-03-06T09:02:36.663522","exception":false,"start_time":"2021-03-06T09:02:35.97757","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:35.926101Z","iopub.execute_input":"2021-11-28T01:10:35.927124Z","iopub.status.idle":"2021-11-28T01:10:36.707783Z","shell.execute_reply.started":"2021-11-28T01:10:35.927068Z","shell.execute_reply":"2021-11-28T01:10:36.706968Z"}}
tigna_plot(df['Volume_(BTC)'], 'Distribution of Volume in BTC ', custom_colors[0])

# %% [code] {"papermill":{"duration":0.732128,"end_time":"2021-03-06T09:02:37.447732","exception":false,"start_time":"2021-03-06T09:02:36.715604","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:36.709028Z","iopub.execute_input":"2021-11-28T01:10:36.70993Z","iopub.status.idle":"2021-11-28T01:10:37.464451Z","shell.execute_reply.started":"2021-11-28T01:10:36.709883Z","shell.execute_reply":"2021-11-28T01:10:37.463591Z"}}
tigna_plot(df['Volume_(Currency)'], 'Distribution of Volume', custom_colors[1])

# %% [code] {"execution":{"iopub.status.busy":"2021-11-28T01:10:37.465668Z","iopub.execute_input":"2021-11-28T01:10:37.466242Z","iopub.status.idle":"2021-11-28T01:10:38.323813Z","shell.execute_reply.started":"2021-11-28T01:10:37.46621Z","shell.execute_reply":"2021-11-28T01:10:38.322841Z"}}
tigna_plot(df['Weighted_Price'], 'Distribution of Weighted price', custom_colors[2])

# %% [code] {"papermill":{"duration":0.39603,"end_time":"2021-03-06T09:02:37.991914","exception":false,"start_time":"2021-03-06T09:02:37.595884","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:10:38.325262Z","iopub.execute_input":"2021-11-28T01:10:38.325501Z","iopub.status.idle":"2021-11-28T01:10:39.34492Z","shell.execute_reply.started":"2021-11-28T01:10:38.325473Z","shell.execute_reply":"2021-11-28T01:10:39.343995Z"}}
plt.figure(figsize=(8, 8))
corr = merged_df[merged_df.columns[1:]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(merged_df[merged_df.columns[1:]].corr(), mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5, annot=True)
plt.savefig('my_image.png')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-28T01:10:39.34624Z","iopub.execute_input":"2021-11-28T01:10:39.346787Z","iopub.status.idle":"2021-11-28T01:10:39.354774Z","shell.execute_reply.started":"2021-11-28T01:10:39.346749Z","shell.execute_reply":"2021-11-28T01:10:39.353701Z"}}
merged_df = merged_df.drop(
    columns=['volume(BTC)/Volume_market_mean', 'Volume_(Currency)/volume_curr_mean', 'close/close_market_mean'])
merged_df.columns

# %% [markdown] {"papermill":{"duration":0.053145,"end_time":"2021-03-06T09:02:38.097689","exception":false,"start_time":"2021-03-06T09:02:38.044544","status":"completed"},"tags":[]}
#
#

# %% [code] {"papermill":{"duration":0.067579,"end_time":"2021-03-06T09:02:38.329964","exception":false,"start_time":"2021-03-06T09:02:38.262385","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:16:53.296822Z","iopub.execute_input":"2021-11-28T01:16:53.297135Z","iopub.status.idle":"2021-11-28T01:16:53.312672Z","shell.execute_reply.started":"2021-11-28T01:16:53.297102Z","shell.execute_reply":"2021-11-28T01:16:53.311975Z"}}
indices = corr.index.values
cor_matrix = np.asmatrix(corr)
G = nx.from_numpy_matrix(cor_matrix)
G = nx.relabel_nodes(G, lambda x: indices[x])


# G.edges(data=True)


def corr_network(g, co_dir, min_cor):
    h = g.copy()

    for i, j, w in g.edges(data=True):
        if co_dir == "positive":
            if w["weight"] < 0 or w["weight"] < min_cor:
                h.remove_edge(i, j)
        else:
            if w["weight"] >= 0 or w["weight"] > min_cor:
                h.remove_edge(i, j)

    e, w = zip(*nx.get_edge_attributes(h, 'weight').items())
    w = tuple([(1 + abs(i)) ** 2 for i in w])

    j = dict(nx.degree(h))
    nodelist = j.keys()
    node_sizes = j.values()

    positions = nx.circular_layout(h)

    plt.figure(figsize=(5, 5))

    nx.draw_networkx_nodes(h, positions, node_color='#d100d1', nodelist=nodelist,
                           node_size=tuple([x ** 2 for x in node_sizes]), alpha=0.8)

    nx.draw_networkx_labels(h, positions, font_size=13)

    if co_dir == "positive":
        edge_colour = plt.cm.summer
    else:
        edge_colour = plt.cm.autumn

    nx.draw_networkx_edges(h, positions, edgelist=e, style='solid',
                           width=w, edge_color=w, edge_cmap=edge_colour,
                           edge_vmin=min(w), edge_vmax=max(w))
    plt.axis("off")
    plt.show()


# %% [code] {"papermill":{"duration":0.192351,"end_time":"2021-03-06T09:02:38.572647","exception":false,"start_time":"2021-03-06T09:02:38.380296","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-11-28T01:16:53.577541Z","iopub.execute_input":"2021-11-28T01:16:53.577988Z","iopub.status.idle":"2021-11-28T01:16:54.310896Z","shell.execute_reply.started":"2021-11-28T01:16:53.577953Z","shell.execute_reply":"2021-11-28T01:16:54.310047Z"}}
corr_network(G, co_dir="positive", min_cor=0.5)

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]
