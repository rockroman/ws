import streamlit as st
from feature_engine.discretisation import ArbitraryDiscretiser
import numpy as np
from src.data_management import load_ames_data
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps
sns.set_style("whitegrid")


def correlation_analysis_page():

    # load data
    df = load_ames_data()

    # correlated study results -> features with high correlation
    vars_to_study = [
        '1stFlrSF',
        'GarageArea',
        'GarageYrBlt',
        'GrLivArea',
        'KitchenQual_Ex',
        'KitchenQual_TA',
        'OverallQual',
        'TotalBsmtSF',
        'YearBuilt',
        'YearRemodAdd'
    ]

    st.header("House price study")
    st.write(
        """
        The client wants to understand how different house attributes are related
        to sale prices.
        As a result, they expect data visualizations showing the correlation
        between these variables and the sale price.
        """
    )

    # inspect data
    if st.checkbox("Inspect house data from the area"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        """
        A correlation study was performed in the notebook to gain a deeper
        understanding of how the variables relate to the house sale price. \n
        The variables most strongly correlated with the sale price are: \n"
        """
        f"**{vars_to_study}**"
    )

    st.info(
        """
        The correlation findings and plot interpretations result in the following conclusions:. \n

        * Larger houses are more valuable: Houses with larger areas across with different additional features show a 
        strong correlation with higher values. This was the most significant correlation found in the analysis.
        Further analysis confirmed that overall quality remains the most important feature when assessing
        feature importance. \n

        * Better condition and higher-quality features increase the value: Houses in better condition and with
        higher-quality building features tend to be more valuable, supporting the first hypothesis. \n

        * Newly built or renovated houses are more valuable: Houses that are newer or recently renovated also
        tend to be have a higher value, supporting and confirming the second hypothesis. \n
        """
    )

    df_eda = df.filter(vars_to_study + ['SalePrice'])
    target_var = 'SalePrice'
    st.write("#### Data visualizations")

    # Plot to display the distribution of the target variable
    if st.checkbox("Distribution of target variable"):
        plot_target_hist(df_eda, target_var)

    # Show heatmaps

    if st.checkbox("Show Correlation and PPS Heatmaps"):
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df_eda)
        DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix, CorrThreshold=0.4, PPS_Threshold=0.2)

    # functions that will lead to a display of plots of the variables

    def scatter_plot_for_eda(df, col, target_var):
        fig = plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x=col, y=target_var)
        plt.title(f"{col}", fontsize=20, y=1.05)
        st.pyplot(fig)

    def plot_categorical(df, col, target_var):
        fig, axes = plt.subplots(figsize=(12, 5))
        sns.countplot(data=df, x=col, hue=target_var,
                      order=df[col].value_counts().index)
        plt.xticks(rotation=90)
        plt.title(f"{col}", fontsize=20, y=1.05)
        st.pyplot(fig)

    def variables_plots(df_eda):
        target_var = 'SalePrice'

        for col in df_eda.drop([target_var], axis=1).columns.to_list():
            if df_eda[col].dtype == 'object':
                plot_categorical(df_eda, col, target_var)
            else:
                scatter_plot_for_eda(df_eda, col, target_var)

    # Plots per variable
    if st.checkbox("Variables Plots - visual analysis"):
        variables_plots(df_eda)


def plot_target_hist(df, target_var):
    """
    Function for the histogram plot of the target variable
    """
    fig, axes = plt.subplots(figsize=(12, 6))
    sns.histplot(data=df, x=target_var, kde=True)
    plt.title(f"Distribution of {target_var}", fontsize=20)
    st.pyplot(fig)


def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
    """
    Function to create heatmap using correlations.
    """
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        fig, axes = plt.subplots(figsize=figsize)
        sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                    mask=mask, cmap='viridis', annot_kws={"size": font_annot}, ax=axes,
                    linewidth=0.5)
        axes.set_yticklabels(df.columns, rotation=0)
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    """
    Function to create heatmap with pps.
    """
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[abs(df) < threshold] = True

        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                         mask=mask, cmap='rocket_r', annot_kws={"size": font_annot},
                         linewidth=0.05, linecolor='grey')

        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def CalculateCorrAndPPS(df):
    """
    Function for calculation of correlations and pps.
    """
    df_corr_spearman = df.corr(method="spearman")
    df_corr_spearman.name = 'corr_spearman'
    df_corr_pearson = df.corr(method="pearson")
    df_corr_pearson.name = 'corr_pearson'

    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(columns='x', index='y', values='ppscore')

    pps_score_stats = pps_matrix_raw.query("ppscore < 1").filter(['ppscore']).describe().T
    print(pps_score_stats.round(3))

    return df_corr_pearson, df_corr_spearman, pps_matrix


def DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix, CorrThreshold, PPS_Threshold,
                      figsize=(20, 12), font_annot=8):
    """
    Function to display the correlations and pps.
    """

    heatmap_corr(df=df_corr_spearman, threshold=CorrThreshold, figsize=figsize, font_annot=font_annot)

    heatmap_corr(df=df_corr_pearson, threshold=CorrThreshold, figsize=figsize, font_annot=font_annot)

    heatmap_pps(df=pps_matrix, threshold=PPS_Threshold, figsize=figsize, font_annot=font_annot)
