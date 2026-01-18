# Surprise Housing – Advanced Regression
Predict house prices using regularised regression based on the Surprise Housing assignment. The notebook implements end‑to‑end data preparation, EDA, feature engineering, and Ridge/Lasso model training and evaluation.


## Table of Contents
* [General Info](#general-information)
* [What's Done](#whats-done)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- **Background:** Predict `SalePrice` for houses using regularised regression to support investment decisions.
- **Business Problem:** Estimate realistic sale prices from available attributes to inform buy/flip strategy.
- **Dataset:** `train.csv` (see `data_description.txt` for variable definitions). Target is `SalePrice`.
- **Notebook:** `advanced_regression.ipynb` contains the complete workflow implemented.

## What's Done
- Libraries: Imported `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`; display settings; `%matplotlib inline`.
- Data loading: Read `train.csv` into `house_df`.
- Missing values: Dropped high‑missing cols (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`); imputed `LotFrontage` (median), `MasVnrArea` (median), `GarageYrBlt` (mean); set "No Garage"/"No Basement" where applicable; `MasVnrType`→"None"; `Electrical`→"SBrkr".
- Categorical tidying: Mapped quality classes; consolidated rare categories into "Others"; dropped skewed/less informative variables (e.g., `Functional`, `GarageQual`, `GarageCond`, `SaleType`, `CentralAir`, `Heating`, `ExterCond`, `RoofMatl`, `Utilities`, `Street`, `Condition1`, `Condition2`, `BsmtFinType2`, `PavedDrive`, `BsmtCond`).
- Target transform: Log‑transformed `SalePrice` and verified near‑normal distribution.
- EDA: Countplots/boxplots of key categorical variables; correlation heatmap for numeric features.
- Feature engineering: Added `YearSinceRemodel`.
- Encoding: Created dummy variables for categorical features; concatenated and dropped original categoricals.
- Cleanup: Dropped `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `YrSold` as non‑relevant.
- Split & scale: Train/test split (70/30, `random_state=100`); scaled selected numerical features via `StandardScaler`.
- Modeling: Ridge and Lasso with `GridSearchCV`; selected `alpha=10` (Ridge) and `alpha=0.001` (Lasso); trained and evaluated on train/test.

## Conclusions
- **Target handling:** Log‑transforming `SalePrice` reduced skew; distribution improved.
- **Models built:** Ridge and Lasso with cross‑validated alphas (Ridge=10, Lasso=0.001).
- **Performance (as per notebook):** Ridge slightly outperformed Lasso; both strong on train/test.
- **Top drivers (from coefficients):** Dummy features for `SaleCondition_*` and `GarageFinish_*` were among the most influential.
- **Recommendation:** Prefer Lasso for future predictions due to built‑in feature selection and comparable accuracy.

## Technologies Used
NumPy: 2.0.2
Pandas: 2.2.2
Matplotlib: 3.10.0
Seaborn: 0.13.2
Scikit-learn: 1.6.1

## Getting Started
- **Clone & setup environment:**
- **Run the notebook:**
- **Data files:** Ensure `train.csv` and `data_description.txt` are present in the project root.

## Acknowledgements
- UpGrad Advanced Regression module (Surprise Housing assignment context).
- scikit‑learn, seaborn, and matplotlib documentation.

## Contact
Created by [@Faseelogic] — PRs and suggestions welcome!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->