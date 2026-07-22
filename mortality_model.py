# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:29:39 2026

@author: dogra
"""

import pandas as pd
import numpy as np
from numpy.linalg import svd
from statsmodels.tsa.arima.model import ARIMA

def load_and_clean_raw_data(file_path: str) -> pd.DataFrame:
    """Load the raw CDC WONDER extract and perform initial processing."""
    df = pd.read_csv(file_path)
    df = df[df["Single-Year Ages Code"].notna()]
    df["Age_Numeric"] = pd.to_numeric(df["Single-Year Ages Code"],errors="coerce")
    
    return df

def create_age_groups(age: int | float) -> str:
    """Map a single age to its 5-year age band label."""
    bins = [0, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, np.inf]
    labels = ["0-14","15-19","20-24","25-29","30-34","35-39","40-44",
              "45-49","50-54","55-59","60-64","65-69","70-74","75+"]
    return pd.cut([age], bins=bins, labels=labels)[0]
    
def process_mort_data(
    df: pd.DataFrame,
    race: str = "Black or African American",
    sex: str = "M",
    start_year: int = 2010,
    end_year: int = 2018
) -> pd.DataFrame:
    """Filter and aggregate data, create raw mortality rates per 100,000.
    Drop age categories "0-14" and "75+" due to data sparsity. """
    df = df.copy()
    df["Age Group"] = df["Age_Numeric"].apply(create_age_groups)

    mort_df = df[['Year','Sex Code', 'Race', 'Age Group','Deaths', 'Population']].copy()

    mort_df = mort_df[
        (mort_df["Race"] == race) & (mort_df["Sex Code"] == sex) &
         (mort_df["Year"].between(start_year, end_year))
    ]

    mort_df = mort_df.drop(columns=["Race", "Sex Code"])
    mort_df = mort_df[~mort_df["Age Group"].isin(["0-14", "75+"])]
    mort_df["Deaths"] = pd.to_numeric(mort_df["Deaths"],errors="coerce")
    mort_df["Population"] = pd.to_numeric(mort_df["Population"],errors="coerce")
    
    mort_df = (
        mort_df.groupby(["Year", "Age Group"], as_index=False)
        .agg({"Deaths": "sum","Population": "sum"})
    )

    mort_df["Year"] = mort_df["Year"].astype("int64")
    mort_df["Deaths"] = mort_df["Deaths"].astype("int64")
    
    mort_df['Mort_Rate'] = mort_df['Deaths'] / mort_df['Population'] * 100000
    
    return mort_df

def create_mortality_matrix(mort_df: pd.DataFrame) -> pd.DataFrame:
    """Create log mortality rates for the Lee-Carter model, guarantees that forecasted mortality remains positive.
    Zero-death cells are set to NaN before taking the log, since log(0) is undefined."""
    mort_df = mort_df.copy()
    mort_df.loc[mort_df["Deaths"]==0,"Mort_Rate"] = np.nan
    mort_df["Log_Mort_Rate"]=np.log(mort_df["Mort_Rate"])
    mort_matrix = (mort_df.pivot(index="Age Group",columns="Year",values="Log_Mort_Rate"))

    return mort_matrix

def svd_decomp(mort_matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.DataFrame]:
    """Demean the log mortality rates before SVD. Missing cells are imputed with each age group's minimum"""
    a_x = mort_matrix.mean(axis=1,skipna=True)
    centered = mort_matrix.sub(a_x,axis=0)
    
    row_min = centered.min(axis=1)
    centered = centered.T.fillna(row_min).T

    U, S, VT = svd(centered, full_matrices = False)
    
    return U, S, VT, a_x, centered

def extract_normalized_parameters(U: np.ndarray, S: np.ndarray,VT: np.ndarray,
    a_x: pd.Series,mort_matrix: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Normalize the b_x and k_t parameters for interpretability, adjust a_x accordingly."""
    
    b_x = U[:,0]
    k_t = S[0] * VT[0,:]
    
    scale = b_x.sum()  
    b_x = b_x / scale
    k_t = k_t * scale
    k_mean = k_t.mean()
    k_t = k_t - k_mean
    
    a_x = a_x + b_x * k_mean

    a_x = pd.Series(a_x,index=mort_matrix.index,name="a_x")
    b_x = pd.Series(b_x,index=mort_matrix.index,name="b_x") 
    k_t_index = pd.to_datetime(mort_matrix.columns.astype(str))
    k_t_index = pd.DatetimeIndex(k_t_index, freq="YS") 
    k_t = pd.Series(k_t,index=k_t_index,name="k_t")
    
    return a_x, b_x, k_t

def arima_forecast(k_t: pd.Series):
    """Fit a random-walk-with-drift ARIMA(0,1,0) model to the k_t index."""
    model = ARIMA(k_t, order = (0,1,0), trend = 't', freq = "YS")
    model_fit = model.fit()
    return model_fit

def predict_mort(model_fit, a_x: pd.Series, b_x: pd.Series) -> pd.Series:
    """Forecast one step ahead and convert back to a mortality rate by age group."""
    k_pred = model_fit.forecast(steps=1)
    k_pred_value = k_pred.iloc[0]
    log_mort_pred = a_x + b_x * k_pred_value
    mort_pred = np.exp(log_mort_pred)
    
    return mort_pred

