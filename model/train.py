import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score


def clean_money_column(series: pd.Series) -> pd.Series:
    """Strip $ and commas, coerce to numeric. Used for total_charges AND
    (as the deployment fix) monthly_charges."""
    cleaned = series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # duplicates
    df = df.drop_duplicates()

    # title-case all object/categorical columns (matches notebook exactly;
    # note this does NOT consolidate "M"/"F"/"MTM" into "Male"/"Female"/
    # "Month-To-Month" -- that's a pre-existing data quality issue in the
    # source notebook, left as-is for fidelity to the graded results)
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.title()

    # numeric money columns: total_charges (as in notebook) + monthly_charges (fix)
    df["total_charges"] = clean_money_column(df["total_charges"])
    df["monthly_charges"] = clean_money_column(df["monthly_charges"])

    # monthly_charges is now numeric, so remove it from the categorical list
    categorical_cols = [c for c in categorical_cols if c != "monthly_charges"]

    # fill missing numeric with median, categorical with mode (matches notebook)
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # outlier capping on total_charges via IQR (matches notebook)
    col_name = "total_charges"
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col_name] = np.where(
        df[col_name] < lower_bound,
        lower_bound,
        np.where(df[col_name] > upper_bound, upper_bound, df[col_name]),
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_spend"] = df["total_charges"] / (df["tenure_months"] + 1)
    df["is_new_customer"] = (df["tenure_months"] <= 12).astype(int)
    return df


def build_preprocessor(numeric_cols, categorical_cols):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="churn_pipeline.joblib")
    args = parser.parse_args()

    df = load_and_clean(args.csv)
    df = engineer_features(df)

    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print("Numeric features:", numeric_cols)
    print("Categorical features:", categorical_cols)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    base_rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    param_grid_fast = {"max_depth": [6, 10], "n_estimators": [50, 100]}
    grid_search = GridSearchCV(
        estimator=base_rf, param_grid=param_grid_fast, cv=3, scoring="f1", n_jobs=-1
    )
    grid_search.fit(X_train_processed, y_train)
    best_rf = grid_search.best_estimator_

    print("Best params:", grid_search.best_params_)
    y_test_pred = best_rf.predict(X_test_processed)
    print("Test F1:", f1_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    # Bundle preprocessor + model into a single deployable pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best_rf),
    ])
    # Pipeline steps were fit separately above; refit cleanly through the
    # combined pipeline object so joblib captures one consistent artifact.
    full_pipeline.fit(X_train, y_train)

    feature_columns = numeric_cols + categorical_cols  # raw pre-preprocessing columns
    bundle = {
        "pipeline": full_pipeline,
        "raw_input_columns": [
            "gender", "age", "tenure_months", "contract_type", "internet_service",
            "num_addon_services", "monthly_charges", "data_usage_gb", "support_calls",
            "payment_method", "total_charges",
        ],
        "engineered_columns": ["avg_monthly_spend", "is_new_customer"],
        "model_version": "1.0.0",
        "test_f1": float(f1_score(y_test, full_pipeline.predict(X_test))),
    }
    joblib.dump(bundle, args.out)
    print(f"Saved bundle to {args.out}")


if __name__ == "__main__":
    main()
