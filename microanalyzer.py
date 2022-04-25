
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr 

gene_data = pd.read_csv("input.csv")
header = gene_data.columns[0:].tolist()
genes = gene_data[header[0]].tolist()
for col in gene_data.iloc[1: , 1:61].columns:
    gene_data[col] = gene_data[col].astype(int)

control = pd.DataFrame(gene_data.iloc[: , 1: 31])
tumor = pd.DataFrame(gene_data.iloc[: , 31: 61])

#preforms pearson correlation for a given gene across all samples in the dataset
def pearsoncorrelation  (gene):
    pearsoncorr_control = np.array(control.loc[genes.index(gene)])
    pearsoncorr_tumor = np.array(tumor.loc[genes.index(gene)])
    print(pearsonr(pearsoncorr_tumor, pearsoncorr_control))


def showheatmap():
    plt.subplots(figsize=(35,15))
    sns.heatmap(gene_data, yticklabels = genes)
    plt.show()

showheatmap()





