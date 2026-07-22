# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:32:42 2026

@author: dogra
"""

import mortality_model as model
import results
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_directory = project_root / "data"
    figure_directory = project_root / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    
    file_path = data_directory / "National 2010-2019.csv"
    df = model.load_and_clean_raw_data(file_path)
    mort_data = model.process_mort_data(df, start_year = 2010, end_year = 2018)
    mort_matrix = model.create_mortality_matrix(mort_data)
    
    U, S, VT, a_x, centred_matrix = model.svd_decomp(mort_matrix)    
    a_x, b_x, k_t = model.extract_normalized_parameters(U, S, VT, a_x, centred_matrix)
    
    model_fit = model.arima_forecast(k_t)
    mort_2019_pred = model.predict_mort(model_fit, a_x, b_x)
    
    df = df.copy()
    mort_data_2019_obs = model.process_mort_data(df, start_year = 2019, end_year = 2019)
    mort_2019_obs = (mort_data_2019_obs.set_index("Age Group")["Mort_Rate"])
       
    comparison, _, _ = results.generate_comparison_df( mort_2019_obs, mort_2019_pred)
    results.plot_observed_vs_predicted(comparison, figure_directory)
    results.plot_agg_mort(mort_data, mort_data_2019_obs, mort_2019_pred, figure_directory)
    results.plot_reconstructed_mort_w_pc(a_x, U, S, VT, mort_matrix, figure_directory)
    
    file_path_1 = data_directory / "us-states.json"
    file_path_2 = data_directory / "National Data by State for 2010_2018.csv"
    results.plot_state_mort_rate_heat_map(figure_directory, file_path_1, file_path_2)
    
if __name__ == "__main__":
    main()