# doe_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import scipy.optimize as opt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ML & Stats
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pyDOE2 import fullfact

# ----------------------------
# Streamlit App Configuration
# ----------------------------
st.set_page_config(
    page_title="📊 DOE + ML + Weibull Dissolution App",
    layout="wide"
)

st.title("🎯 Design of Experiments (DOE) & Analysis App")
st.markdown(
    """
    This app supports:
    - DOE design generation (Full Factorial)
    - Response entry (manual or Excel upload)
    - Model fitting with ANOVA and ML (Random Forest, XGBoost)
    - Interactive contour and 3D surface plots
    - Weibull fitting for multipoint dissolution data
    """
)

# ----------------------------
# Section 1: DOE Design Inputs
# ----------------------------
st.header("🛠️ Step 1: Define Factors and Levels for DOE Design")

num_factors = st.number_input("Number of Factors", min_value=1, max_value=5, value=2, step=1)

factor_names = []
factor_levels_list = []

for i in range(num_factors):
    st.subheader(f"Factor {i+1}")
    name = st.text_input(f"Name of Factor {i+1}", value=f"X{i+1}", key=f"fname_{i}")
    levels_str = st.text_area(
        f"Enter levels for {name} (comma separated)", 
        value="0,1", key=f"levels_{i}", height=50
    )
    # Process levels to float list
    try:
        levels = [float(x.strip()) for x in levels_str.split(",") if x.strip() != ""]
    except Exception:
        levels = []
    factor_names.append(name)
    factor_levels_list.append(levels)

# Button to generate DOE
if st.button("Generate DOE Design"):
    # Validate all factors have levels
    if any(len(lvls) == 0 for lvls in factor_levels_list):
        st.error("Please enter at least one level per factor.")
    else:
        # Full factorial design using pyDOE2
        level_counts = [len(lvls) for lvls in factor_levels_list]
        design_indices = fullfact(level_counts).astype(int)

        # Map design indices to actual factor levels
        design_df = pd.DataFrame()
        for i, lvls in enumerate(factor_levels_list):
            design_df[factor_names[i]] = [lvls[idx] for idx in design_indices[:, i]]

        st.session_state["doe_data"] = design_df
        st.session_state["factor_names"] = factor_names

        st.success("DOE design generated!")
        st.dataframe(design_df)

# ----------------------------
# Section 2: Response Entry / Upload
# ----------------------------
st.header("📥 Step 2: Enter or Upload Response Data")

if "doe_data" in st.session_state:
    design_df = st.session_state["doe_data"]
    response_col = st.text_input("Response variable name", value="Y")

    st.subheader("Enter Response Values Manually")
    response_values = []
    for i in range(len(design_df)):
        val = st.number_input(f"Response for run {i+1}", key=f"resp_{i}")
        response_values.append(val)

    if st.button("Submit Response"):
        response_series = pd.Series(response_values, name=response_col)
        st.session_state["response"] = response_series
        st.success("Response data saved!")

    st.subheader("Or Upload Response Data Excel File")
    uploaded_file = st.file_uploader("Upload Excel file (.xlsx or .xls)", type=["xlsx", "xls"])
    if uploaded_file:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(df_uploaded.head())
            if response_col in df_uploaded.columns:
                st.session_state["response"] = df_uploaded[response_col]
                st.success(f"Response column '{response_col}' loaded from file!")
            else:
                st.warning(f"Response column '{response_col}' not found in uploaded file.")
        except Exception as e:
            st.error(f"Error reading Excel: {e}")
else:
    st.info("Generate a DOE design first to enter or upload response data.")

# ----------------------------
# Section 3: Model Fitting (ANOVA + ML)
# ----------------------------
st.header("📈 Step 3: Model Fitting and Analysis")

if "doe_data" in st.session_state and "response" in st.session_state:
    X = st.session_state["doe_data"]
    y = st.session_state["response"]

    st.write("Factors (X):")
    st.dataframe(X)
    st.write("Response (Y):")
    st.write(y)

    # Build formula for ANOVA with main effects and two-factor interactions
    factors = st.session_state["factor_names"]
    formula_terms = factors.copy()
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            formula_terms.append(f"{factors[i]}:{factors[j]}")

    formula = f"{y.name} ~ " + " + ".join(formula_terms)

    df_model = X.copy()
    df_model[y.name] = y

    # Fit OLS model for ANOVA
    try:
        model = ols(formula, data=df_model).fit()
        st.subheader("ANOVA Table")
        anova_results = sm.stats.anova_lm(model, typ=2)
        st.dataframe(anova_results)
    except Exception as e:
        st.error(f"Error fitting ANOVA model: {e}")

    # Fit ML models
    st.subheader("ML Regression Models (Random Forest & XGBoost)")

    try:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        rf_score = rf_model.score(X, y)

        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        xgb_model.fit(X, y)
        xgb_score = xgb_model.score(X, y)

        st.write(f"Random Forest R² Score: {rf_score:.3f}")
        st.write(f"XGBoost R² Score: {xgb_score:.3f}")

        # Save models in session state for later use in plots
        st.session_state["rf_model"] = rf_model
        st.session_state["xgb_model"] = xgb_model

    except Exception as e:
        st.error(f"Error training ML models: {e}")

else:
    st.info("Generate DOE and enter response data first.")

# ----------------------------
# Section 4: Interactive Contour & 3D Surface Plots
# ----------------------------
st.header("📊 Step 4: Interactive Contour & 3D Surface Plots")

if (
    "doe_data" in st.session_state 
    and "response" in st.session_state 
    and "rf_model" in st.session_state
):
    X = st.session_state["doe_data"]
    y = st.session_state["response"]
    rf_model = st.session_state["rf_model"]

    n_factors = len(st.session_state["factor_names"])

    if n_factors not in [2, 3]:
        st.warning("Contour and 3D surface plots require exactly 2 or 3 factors.")
    else:
        factors = st.session_state["factor_names"]

        if n_factors == 3:
            selected_factors = st.multiselect("Select exactly 2 factors to plot", factors, default=factors[:2])
            if len(selected_factors) != 2:
                st.warning("Please select exactly 2 factors to plot.")
            else:
                x_factor, y_factor = selected_factors
        else:
            x_factor, y_factor = factors[0], factors[1]

        if 'x_factor' in locals() and 'y_factor' in locals():
            plot_df = X[[x_factor, y_factor]].copy()
            plot_df['Response'] = y

            # Contour Plot using Plotly
            fig_contour = px.density_contour(
                plot_df, x=x_factor, y=y_factor, z='Response',
                title="Contour Plot of Response", nbinsx=30, nbinsy=30
            )
            fig_contour.update_traces(contours_coloring="fill", contours_showlabels=True)
            st.plotly_chart(fig_contour)

            # 3D Surface Plot preparation
            xi = np.linspace(plot_df[x_factor].min(), plot_df[x_factor].max(), 30)
            yi = np.linspace(plot_df[y_factor].min(), plot_df[y_factor].max(), 30)
            xi, yi = np.meshgrid(xi, yi)

            grid_points = pd.DataFrame({x_factor: xi.ravel(), y_factor: yi.ravel()})

            if n_factors == 3:
                fixed_factor = list(set(factors) - set([x_factor, y_factor]))[0]
                median_val = X[fixed_factor].median()
                grid_points[fixed_factor] = median_val

            # Predict response on grid using Random Forest
            try:
                z_pred = rf_model.predict(grid_points)
            except Exception:
                z_pred = np.zeros_like(xi.ravel())

            zi = z_pred.reshape(xi.shape)

            fig_surface = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi)])
            fig_surface.update_layout(
                title="3D Surface Plot",
                scene=dict(
                    xaxis_title=x_factor,
                    yaxis_title=y_factor,
                    zaxis_title="Response"
                )
            )
            st.plotly_chart(fig_surface)

else:
    st.info("Complete DOE generation, response entry, and model fitting first.")

# ----------------------------
# Section 5: Weibull Fitting for Multipoint Dissolution
# ----------------------------
st.header("💊 Step 5: Weibull Model Fitting for Multipoint Dissolution")

uploaded_wb = st.file_uploader("Upload Multipoint Dissolution Data (Excel)", type=["xlsx", "xls"], key="weibull_upload")

if uploaded_wb:
    try:
        wb_df = pd.read_excel(uploaded_wb)
        st.write("Dissolution Data Preview")
        st.dataframe(wb_df.head())

        time_col = st.selectbox("Select Time Column", wb_df.columns, key="time_col")
        dissolution_col = st.selectbox("Select Dissolution % Column", wb_df.columns, key="diss_col")

        time_data = wb_df[time_col].values
        dissolution_data = wb_df[dissolution_col].values

        def weibull_model(t, alpha, beta):
            return 100 * (1 - np.exp(-(t / alpha) ** beta))

        popt, pcov = opt.curve_fit(weibull_model, time_data, dissolution_data, bounds=(0, np.inf))

        st.write(f"Fitted Weibull Parameters:\n- alpha (scale): {popt[0]:.3f}\n- beta (shape): {popt[1]:.3f}")

        # Plot observed vs fitted
        t_fit = np.linspace(time_data.min(), time_data.max(), 100)
        y_fit = weibull_model(t_fit, *popt)

        fig, ax = plt.subplots()
        ax.plot(time_data, dissolution_data, 'o', label="Observed")
        ax.plot(t_fit, y_fit, '-', label="Weibull Fit")
        ax.set_xlabel("Time")
        ax.set_ylabel("% Dissolution")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in Weibull fitting: {e}")

else:
    st.info("Upload multipoint dissolution data for Weibull model fitting.")

# ----------------------------
# End of app
# ----------------------------

