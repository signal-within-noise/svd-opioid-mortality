# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:30:46 2026

@author: dogra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
from pathlib import Path

def generate_comparison_df(observed: pd.Series, predicted: pd.Series) -> tuple[pd.DataFrame, float, float]:
    """Build an observed-vs-predicted comparison table for 2019 mortality with error metrics."""
    
    comparison = pd.DataFrame({"Observed": observed,"Predicted": predicted})  
    comparison["Error"] = comparison["Predicted"] - comparison["Observed"] 
    comparison["Absolute Error"] = comparison["Error"].abs()
    comparison["Percent Error"] = comparison["Error"] / comparison["Observed"] * 100
   
    mae = comparison["Absolute Error"].mean()
    rmse = np.sqrt(np.mean(comparison["Error"]**2))
    
    return comparison, mae, rmse 

def plot_observed_vs_predicted(comparison: pd.DataFrame,directory: Path,
                               figure_name: str = "observed_vs_predicted_mortality_2019.png") -> None:
    """Plot observed vs. forecasted 2019 mortality by age group."""

    plt.close("all")    
    plt.figure(figsize=(10,6))
    plt.plot(comparison.index,comparison["Observed"],marker="o",linewidth=2.5,markersize=7,label="Observed")
    plt.plot(comparison.index,comparison["Predicted"],marker="s",linewidth=2.5,markersize=7,label="Forecast")
    
    plt.xlabel("Age Group")
    plt.ylabel("Mortality Rate per 100,000")
    plt.title("Observed vs Forecasted 2019 Opioid Overdose Mortality for Black Males by Age Group")
    
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=False)
    
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(directory / figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    
def plot_agg_mort(mort_data: pd.DataFrame,mort_data_2019_obs: pd.DataFrame,mort_2019_pred: pd.Series,
                  directory: Path,figure_name: str = "aggregate_mortality_2010_2019.png") -> None:
    """Plot the population-weighted aggregate mortality rates for 2010-2018, along with the projected and 
    observed rates for 2019."""  
    
    plt.close("all")
    agg_mort = (mort_data.groupby("Year").agg(Deaths=("Deaths","sum"),Population=("Population","sum")))
    agg_mort["Mortality"] = (agg_mort["Deaths"] / agg_mort["Population"] * 100000)
    
    agg_mort_2019 = mort_data_2019_obs["Deaths"].sum() / mort_data_2019_obs["Population"].sum() * 100000
    
    agg_mort_2019_pred = (
        mort_2019_pred * mort_data_2019_obs.set_index("Age Group")["Population"]
        ).sum() / mort_data_2019_obs["Population"].sum()
    
    plt.figure(figsize=(9,5))

    plt.plot(
        agg_mort.index, agg_mort["Mortality"], marker="o",
        linewidth=2.5, label="Observed 2010-2018"
    )
    
    plt.plot(
        [2018, 2019],[agg_mort.loc[2018, "Mortality"], agg_mort_2019_pred],
        linestyle="--", color = "tab:orange",linewidth=2.5,
    )
    
    plt.plot(
        [2018, 2019],[agg_mort.loc[2018, "Mortality"], agg_mort_2019],
        linestyle="--", color = "tab:blue", linewidth=2.5,
    )

    plt.scatter(
        2019, agg_mort_2019_pred, marker="s",
        color = "tab:orange", s=80, label="Forecast 2019"
    )
    
    plt.scatter(
        2019, agg_mort_2019, marker="s",
        color = "tab:blue", s=80, label="Observed 2019"
    )
    
    plt.xlabel("Year")
    plt.ylabel("Mortality Rate per 100,000")
    plt.title("Black Male Opioid Mortality Rate: 2010-2019")
    
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=False)
    
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.xticks(
         ticks=range(2010, 2020),rotation=45,ha="right"
     )
    
    plt.tight_layout()
    plt.savefig(directory / figure_name, dpi=300, bbox_inches="tight")
    plt.close()
    
def plot_reconstructed_mort_w_pc(
    a_x: pd.Series,U: np.ndarray,S: np.ndarray,VT: np.ndarray,mort_matrix: pd.DataFrame,
    directory: Path,figure_name: str = "svd_reconstruction_comparison.png") -> None:
    """Plot observed mortality curves against SVD-reconstructed mortality curves (using both PC1 as 
    well as PC1 & PC2)."""
    
    plt.close("all")
    recon_pc1 = (a_x.values[:, None] + np.outer(U[:,0] * S[0], VT[0,:]))  
    recon_pc2 = (recon_pc1 +np.outer(U[:,1] * S[1], VT[1,:]))
    
    recon_pc1 = np.exp(recon_pc1)
    recon_pc2 = np.exp(recon_pc2)
    
    actual = np.exp(mort_matrix.values)
    
    years = [2010, 2013, 2016, 2018]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    
    for ax, year in zip(axes.flat, years):
        col = mort_matrix.columns.get_loc(year)  
        ax.plot(mort_matrix.index,actual[:, col], marker="o",linewidth=2,label="Observed")
        ax.plot(mort_matrix.index, recon_pc1[:, col],"--s",linewidth=2,label="PC1")
        ax.plot(mort_matrix.index,recon_pc2[:, col],":^",linewidth=2,label="PC1 + PC2")
    
        ax.set_title(str(year))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    
    fig.suptitle(
        "Actual vs. SVD Reconstruction of Black Male Opioid Overdoes Mortality by Age Group for Select Years",
        fontsize=16,y=0.98
    )
    fig.supxlabel("Age Group")
    fig.supylabel("Mortality Rate per 100,000")
    
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,bbox_to_anchor=(0.5, 0.92))
    
    plt.tight_layout(rect=[0,0,1,0.90])
    plt.savefig(directory / figure_name, dpi=300, bbox_inches="tight")
    plt.close()

def plot_state_mort_rate_heat_map(directory: Path, state_map_file: str, mortality_file: str,
    year_1: int = 2010,year_2: int = 2018,figure_name: str = "state_mortality_heatmap_2010_2018.png"
):
    """Plot side-by-side geographical heat maps of state-level opioid mortality rates (per 100,000).
    Merge mortality data with geographic boundary data at the state-level."""
    
    plt.close("all")
    states = gpd.read_file(state_map_file)
    state_abbrev = {
        "Alabama": "AL","Arizona": "AZ","Arkansas": "AR","California": "CA","Colorado": "CO","Connecticut": "CT",
        "Delaware": "DE","Florida": "FL","Georgia": "GA","Idaho": "ID","Illinois": "IL","Indiana": "IN",
        "Iowa": "IA","Kansas": "KS","Kentucky": "KY","Louisiana": "LA","Maine": "ME","Maryland": "MD",
        "Massachusetts": "MA","Michigan": "MI","Minnesota": "MN","Mississippi": "MS","Missouri": "MO",
        "Montana": "MT","Nebraska": "NE","Nevada": "NV","New Hampshire": "NH","New Jersey": "NJ",
        "New Mexico": "NM","New York": "NY","North Carolina": "NC","North Dakota": "ND","Ohio": "OH",
        "Oklahoma": "OK","Oregon": "OR","Pennsylvania": "PA","Rhode Island": "RI","South Carolina": "SC",
        "South Dakota": "SD","Tennessee": "TN","Texas": "TX","Utah": "UT","Vermont": "VT","Virginia": "VA","Washington": "WA",
        "West Virginia": "WV","Wisconsin": "WI","Wyoming": "WY", "District of Columbia": "DC"
    }
    
    states["abbr"] = states["name"].map(state_abbrev)
    states["label_point"] = states.geometry.representative_point()
    states = states[~states["name"].isin(["Alaska", "Hawaii"])]
    
    df = pd.read_csv(mortality_file)
    mort_df = df[['Year','State','Deaths', 'Population']].copy()
    mort_df["Population"] = pd.to_numeric(mort_df["Population"])
    mort_df["Deaths"] = pd.to_numeric(mort_df["Deaths"])
    mort_df['Mort_Rate'] = mort_df['Deaths'] / mort_df['Population'] * 100000
    
    mort_year_1 = mort_df[mort_df["Year"] == year_1]
    mort_year_2 = mort_df[mort_df["Year"] == year_2]
    
    map_year_1 = states.merge(mort_year_1,left_on="name",right_on="State",how = "left")  
    map_year_2 = states.merge(mort_year_2,left_on="name",right_on="State",how = "left")
    
    vmin = mort_df["Mort_Rate"].min()
    vmax = mort_df["Mort_Rate"].max()
    
    norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = "YlOrRd"
    
    fig, axes = plt.subplots(1,2,figsize=(13, 5.5))
 
    plot_kwargs = {
        "column": "Mort_Rate","cmap": cmap,"norm": norm,"edgecolor": "gray",
        "linewidth": 0.25,"missing_kwds": {"color": "lightgrey"
        }
    }
    
    map_year_1.plot(ax=axes[0],**plot_kwargs) 
    axes[0].set_title("2010",fontsize=14)
    map_year_2.plot(ax=axes[1],**plot_kwargs)
    axes[1].set_title("2018",fontsize=14)
    
    label_offsets = {
        "RI": (3.5, 0.2),"CT": (3.5, 0.8),"NJ": (3.5, -0.3),
        "DE": (3.5, -1.0),"MD": (3.5, -1.5),"MA": (3.5, 1.2),
    }
    
    exclude_labels = {"DC"}
        
    for ax, map_df in zip(axes,[map_year_1, map_year_2]):
        for _, row in map_df.iterrows():
            state = row["abbr"]
            if state in exclude_labels:
                continue
    
            point = row["label_point"]
            state = row["abbr"]
    
            if state in label_offsets:
                dx, dy = label_offsets[state]
                ax.annotate(
                    state, xy=(point.x, point.y),xytext=(point.x + dx,point.y + dy),
                    fontsize=7,ha="center",va="center",
                    arrowprops={"arrowstyle": "-","linewidth": 0.5}
                )
    
            else:
                ax.text(point.x,point.y,state,fontsize=7,ha="center",va="center")
    
    for ax in axes:
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.axis("off")
    
    fig.subplots_adjust(left=0.02,right=0.85,bottom=0.05,top=0.85,wspace=0.02)

    sm = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
    sm.set_array([])
    
    cbar_ax = fig.add_axes([0.88, 0.25, 0.025, 0.5])
    cbar = fig.colorbar(sm,cax=cbar_ax)
    cbar.set_label("Opioid Mortality Rate (per 100,000)",fontsize=11)
 
    fig.suptitle(
        "Black Male Opioid Mortality Rates per 100,000 by State (Contiguous U.S.), 2010 vs. 2018",
        fontsize=16,y=0.95
    )
    
    plt.savefig(directory / figure_name, dpi=300, bbox_inches="tight")
    plt.close()

    