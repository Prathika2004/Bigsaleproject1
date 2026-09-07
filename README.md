# Big Sales Prediction

## Problem Statement
Retailers running large-scale sales events need to forecast expected item-level sales to plan inventory and staffing, but sales depend on a mix of product attributes (weight, visibility, price) and outlet attributes (size, location, type) that interact in non-obvious ways.

## What This Project Solves
Predicts `Item_Outlet_Sales` (the sales value for a given product at a given outlet) from product and outlet features, using a Random Forest regressor.

## Approach
- **Data cleaning:** imputes missing `Item_Weight` values using the median weight per `Item_Type`; removes outliers in the target variable using a z-score threshold.
- **Categorical encoding:** normalizes inconsistent category labels (e.g. `LF`/`low fat`/`Low Fat` all mapped to one value), then encodes `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, and `Outlet_Type` into numeric codes.
- **Correlation analysis:** checks pairwise correlation between features and the sales target after encoding.
- **Model:** `RandomForestRegressor` (scikit-learn defaults), evaluated with mean absolute error and an actual-vs-predicted scatter plot on a held-out test split.

## Tech Stack
Python, pandas, NumPy, scikit-learn, SciPy, seaborn/matplotlib. Built and run in Google Colab.

## How to Run
Open `Bigsale.ipynb` in Google Colab (or Jupyter) and run all cells. The notebook prompts for a file upload (`Big Sales Data.csv`) via `google.colab.files.upload()` - the dataset itself isn't included in this repo, so you'll need your own copy of the Big Mart / Big Sales dataset with matching columns (`Item_Identifier`, `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`, `Outlet_Identifier`, `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`, `Item_Outlet_Sales`).

## Notes
- No hyperparameter tuning is currently performed - `RandomForestRegressor()` is trained with default parameters.
- Evaluation is limited to MAE; no cross-validation or comparison against other model types.
