import pandas as pd
import numpy as np


CONTINUOUS_FEATURES = [
    'LotFrontage',
    'LotArea',
    'YearBuilt',
    'YearRemodAdd',
    'MasVnrArea',
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'GrLivArea',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'GarageYrBlt',
    'GarageCars',
    'GarageArea',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
    'YrSold'
]


LABEL_NAME = 'SalePrice'


def get_input():
    """do something
    """

    df = pd.read_csv('/home/anish/PycharmProjects/kaggle/house_price/house_price/dat/train.csv').set_index('Id')
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]

    y_mean = df[target_col].mean()

    df = _clean_df(df)

    label_map = get_label_maps(df)

    x = df_to_x(df, label_map, feature_cols, y_mean)

    x = np.array(x, dtype='float32')

    y = np.array(df[target_col], dtype='float32')

    return x, y


def df_to_x(df, label_map, feature_cols, y_mean):
    X = []

    for r in df.values.tolist():
        r = r[:-1]
        x = [0] * len(r)
        for i in range(len(r)):
            c = feature_cols[i]

            if c not in CONTINUOUS_FEATURES:
                try:
                    x[i] = label_map[c][r[i]]

                except KeyError:
                    x[i] = y_mean

            else:
                x[i] = r[i]

        X.append(x)

    return X


def _clean_df(df):
    for c in CONTINUOUS_FEATURES:
        m = df[c].fillna(np.nan).mean()
        df[c] = df[c].fillna(m)

    df = df.fillna('NA')
    return df


def get_label_maps(d):
    cols = d.columns[:-1]

    label_maps = {}

    for c in cols:
        if c not in CONTINUOUS_FEATURES:
            df_map = d[[c, LABEL_NAME]].groupby(c).mean()
            dico_map = {}

            for ix in df_map.index:
                dico_map[ix] = float(df_map[LABEL_NAME].loc[ix])

            label_maps[c] = dico_map

    return label_maps


def main():
    d = get_input()
    print(d)


if __name__ == '__main__':
    main()

