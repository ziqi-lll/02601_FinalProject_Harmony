# Project Data

This project uses liver single-cell RNA sequencing data available
through the HuBMAP Data Portal.

## Data Source

[View the source datasets on HuBMAP](https://portal.hubmapconsortium.org/search/datasets?data_product=6327fd11-974c-48d1-910d-9b981a84c28e)

The local file `LV_raw.h5ad` contains 7,916 cells and 60,286 genes
from the following five datasets:

- HBM254.SJWB.835
- HBM355.DGKN.367
- HBM636.CHVG.283
- HBM947.MQZT.787
- HBM954.WWJF.865

Its expression matrix was verified to match the combined `X` matrices
of the five corresponding downloaded expression files after aligning
cells by dataset and barcode.

## Local Setup

Place the combined data file at:

    data/LV_raw.h5ad

The combined file is not included in this repository because it exceeds
the 25 MB browser upload limit shown by GitHub.

## Analysis

The file contains expression data and cell metadata, including dataset
labels. It does not contain PCA coordinates. Preprocessing and PCA are
performed before Harmony batch correction.
