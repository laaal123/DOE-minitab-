# app.py
import streamlit as st
st.set_page_config(page_title="Pharma DOE & RSM — Polished Lab App", layout="wide")

# Standard imports
import pandas as pd
import numpy as np
import json, os, io, hashlib, datetime
from pyDOE2 import fullfact, ccdesign, bbdesign
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.optimize import curve_fit
from sklearn.utils import resample
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# File structure & utils
# ---------------------------
USERS_FILE = "users.json"
PROJECTS_DIR = "projects"
os.makedirs(PROJECTS_DIR, exist_ok=True)

def sha256(s: str):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def ensure_user_folder(username):
    path = os.path.join(PROJECTS_DIR, username)
    os.makedirs(path, exist_ok=True)
    return path

def save_project(username, name, payload: dict):
    path = ensure_user_folder(username)
    fname = os.path.join(path, f"{name}.json")
    with open(fname, 'w') as f:
        json.dump(payload, f, default=str, indent=2)
    return fname

def load_project(username, name):
    path = ensure_user_folder(username)
    fname = os.path.join(path, f"{name}.json")
    if not os.path.exists(fname):
        return None
    with open(fname, 'r') as f:
        return json.load(f)

def list_projects(username):
    path = ensure_user_folder(username)
    return [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.json')]

# ---------------------------
# Modeling helpers
# ---------------------------
def map_design_to_real(design, bounds, doe_type):
    df = pd.DataFrame(design, columns=[f"F{i+1}" for i in range(design.shape[1])])
    for i, (low, high) in enumerate(bounds):
        if doe_type == "Full Factorial":
            df.iloc[:, i] = df.iloc[:, i].apply(lambda x: low + (high - low) * x)
        else:
            df.iloc[:, i] = df.iloc[:, i].apply(lambda x: ((x + 1) / 2) * (high - low) + low)
    return df

def real_to_coded(row, bounds):
    coded = []
    for i, (low, high) in enumerate(bounds):
        center = (low + high) / 2.0
        half_range = (high - low) / 2.0
        coded_val = (row[i] - center) / half_range
        coded.append(coded_val)
    return np.array(coded)

def build_coded_dataframe(df_real, bounds, factor_names):
    df = df_real.copy().reset_index(drop=True)
    for i, name in enumerate(factor_names):
        center = (bounds[i][0] + bounds[i][1]) / 2.0
        half_range = (bounds[i][1] - bounds[i][0]) / 2.0
        df[f"c_{name}"] = (df[name] - center) / half_range
    return df

def polynomial_terms_coded(df_coded, factor_names):
    df_terms = df_coded.copy()
    for i in range(len(factor_names)):
        for j in range(i+1, len(factor_names)):
            a = f"c_{factor_names[i]}"
            b = f"c_{factor_names[j]}"
            df_terms[f"{a}_x_{b}"] = df_terms[a] * df_terms[b]
    for name in factor_names:
        a = f"c_{name}"
        df_terms[f"{a}__2"] = df_terms[a]**2
    return df_terms

# Weibull
def weibull(t, A, k, b):
    return A * (1 - np.exp(-(t / k)**b))

def fit_weibull(times, y):
    try:
        p0 = [min(100, max(y)*1.05), max(1.0, np.median(times)+1), 1.0]
        popt, _ = curve_fit(weibull, times, y, p0=p0, maxfev=40000)
        return popt.tolist()
    except Exception:
        return [np.nan, np.nan, np.nan]

# Bootstrap PI
def bootstrap_prediction_intervals(formula, df, Xnew_df, n_boot=1000, alpha=0.05, random_state=0):
    rng = np.random.RandomState(random_state)
    preds = []
    for i in range(n_boot):
        sample = resample(df, replace=True, n_samples=len(df), random_state=rng.randint(0, 1e9))
        try:
            m = smf.ols(formula, data=sample).fit()
            p = m.predict(Xnew_df)[0]
        except Exception:
            p = np.nan
        preds.append(p)
    arr = np.array(preds)
    lower = np.nanpercentile(arr, 100*alpha/2.0)
    upper = np.nanpercentile(arr, 100*(1-alpha/2.0))
    median = np.nanpercentile(arr, 50)
    return median, lower, upper, arr

# Desirability helpers
def desirability_maximize(y, L, T, s=1.0):
    if np.isnan(y): return 0.0
    if y <= L: return 0.0
    if y >= T: return 1.0
    return ((y - L) / (T - L))**s

def desirability_minimize(y, L, T, s=1.0):
    if np.isnan(y): return 0.0
    if y <= T: return 1.0
    if y >= L: return 0.0
    return ((L - y) / (L - T))**s

# ---------------------------
# Authentication UI (simple)
# ---------------------------
st.sidebar.header("Account")
users = load_users()
if 'username' not in st.session_state:
    st.session_state['username'] = None

if st.session_state['username'] is None:
    auth_mode = st.sidebar.selectbox("Auth action", ["Login", "Register"])
    if auth_mode == "Register":
        st.sidebar.markdown("**Create account**")
        new_user = st.sidebar.text_input("Username")
        new_pw = st.sidebar.text_input("Password", type="password")
        confirm_pw = st.sidebar.text_input("Confirm password", type="password")
        if st.sidebar.button("Create account"):
            if not new_user or not new_pw:
                st.sidebar.error("Provide username and password.")
            elif new_user in users:
                st.sidebar.error("User exists.")
            elif new_pw != confirm_pw:
                st.sidebar.error("Passwords do not match.")
            else:
                users[new_user] = {"pw": sha256(new_pw)}
                save_users(users)
                ensure_user_folder(new_user)
                st.sidebar.success("User created. Please login.")
    else:
        st.sidebar.markdown("**Login**")
        lu = st.sidebar.text_input("Username")
        lp = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if lu in users and users[lu]["pw"] == sha256(lp):
                st.session_state['username'] = lu
                st.sidebar.success(f"Logged in as {lu}")
            else:
                st.sidebar.error("Invalid credentials.")
else:
    st.sidebar.success(f"Signed in: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

if st.session_state['username'] is None:
    st.title("Pharma DOE & RSM — Polished App")
    st.info("Please create an account or login from the sidebar to proceed.")
    st.stop()

# ---------------------------
# Main UI header
# ---------------------------
st.title("💊 Pharma DOE & RSM — Polished Lab App")
st.markdown("Create DOE → Enter results → Fit coded RSM → Bootstrap PIs → Weibull profiles → Optimize & export.")

# Projects in sidebar
st.sidebar.header("Projects")
proj_action = st.sidebar.selectbox("Project action", ["New project", "Load project", "Delete project"])
user_projects = list_projects(st.session_state['username'])
if proj_action == "New project":
    new_pname = st.sidebar.text_input("Project name")
    if st.sidebar.button("Create new project"):
        if not new_pname:
            st.sidebar.error("Provide project name.")
        else:
            st.session_state['project_name'] = new_pname
            st.session_state['design_df'] = None
            st.session_state['data_df'] = None
            st.session_state['models'] = {}
            st.sidebar.success(f"Project '{new_pname}' created in session.")
elif proj_action == "Load project":
    chosen = st.sidebar.selectbox("Select project", options=[""] + user_projects)
    if chosen and st.sidebar.button("Load"):
        payload = load_project(st.session_state['username'], chosen)
        if payload:
            st.session_state['project_name'] = chosen
            st.session_state['factor_names'] = payload.get('factor_names')
            st.session_state['bounds'] = payload.get('bounds')
            st.session_state['design_df'] = pd.read_json(payload['design_df']) if payload.get('design_df') else None
            st.session_state['data_df'] = pd.read_json(payload['data_df']) if payload.get('data_df') else None
            st.session_state['models'] = {}
            st.sidebar.success(f"Project '{chosen}' loaded.")
        else:
            st.sidebar.error("Failed to load project.")
elif proj_action == "Delete project":
    chosen = st.sidebar.selectbox("Select project to delete", options=[""] + user_projects, key="del_proj")
    if chosen and st.sidebar.button("Delete"):
        path = os.path.join(PROJECTS_DIR, st.session_state['username'], f"{chosen}.json")
        try:
            os.remove(path)
            st.sidebar.success("Deleted.")
        except Exception as e:
            st.sidebar.error(f"Delete failed: {e}")

# ---------------------------
# DOE Setup (sidebar)
# ---------------------------
st.sidebar.header("DOE / Factors Setup")
doe_type = st.sidebar.selectbox("Design type", ["Full Factorial", "Central Composite (CCD)", "Box-Behnken"])
n_factors = st.sidebar.number_input("Number of factors", min_value=2, max_value=8, value=3, step=1)

# load previous if exist
factor_names = st.session_state.get('factor_names', [f"Factor{i+1}" for i in range(int(n_factors))])
bounds = st.session_state.get('bounds', [(0.0, 1.0) for _ in range(int(n_factors))])

new_factor_names = []
new_bounds = []
for i in range(int(n_factors)):
    col1, col2 = st.sidebar.columns([2,2])
    with col1:
        name = st.text_input(f"Name {i+1}", value=(factor_names[i] if i < len(factor_names) else f"Factor{i+1}"), key=f"name_main_{i}")
    with col2:
        low = st.sidebar.number_input(f"{name} low", value=(bounds[i][0] if i < len(bounds) else 0.0), key=f"low_main_{i}")
        high = st.sidebar.number_input(f"{name} high", value=(bounds[i][1] if i < len(bounds) else 1.0), key=f"high_main_{i}")
    new_factor_names.append(name)
    new_bounds.append((float(low), float(high)))

factor_names = new_factor_names
bounds = new_bounds
st.session_state['factor_names'] = factor_names
st.session_state['bounds'] = bounds

if doe_type == "Central Composite (CCD)":
    center_pts = st.sidebar.number_input("Center point repeats", min_value=0, max_value=10, value=4)

if st.sidebar.button("Generate DOE"):
    if doe_type == "Full Factorial":
        design = fullfact([2] * int(n_factors))
    elif doe_type == "Central Composite (CCD)":
        design = ccdesign(int(n_factors), center=(center_pts, center_pts))
    else:
        design = bbdesign(int(n_factors))
    design_df = map_design_to_real(design, bounds, doe_type)
    design_df.columns = factor_names
    design_df = design_df.reset_index(drop=True)
    st.session_state['design_df'] = design_df
    st.success(f"DOE generated ({len(design_df)} runs).")

# ---------------------------
# Main: Design & Data
# ---------------------------
st.header("Design & Data")
colA, colB = st.columns([1,1])

with colA:
    st.subheader("DOE Matrix")
    if 'design_df' in st.session_state and st.session_state['design_df'] is not None:
        st.dataframe(st.session_state['design_df'].round(5))
    else:
        st.info("Generate DOE from the sidebar.")

with colB:
    st.subheader("Upload / Create Data")
    uploaded = st.file_uploader("Upload wide-format CSV/XLSX (factors + responses D5,D15,...) ", type=['csv','xlsx'])
    if uploaded:
        if uploaded.name.endswith('.csv'):
            df_up = pd.read_csv(uploaded)
        else:
            df_up = pd.read_excel(uploaded)
        st.session_state['data_df'] = df_up
        st.success("Uploaded and loaded into session.")
        st.dataframe(df_up.head(10))
    if st.button("Create editable data from design"):
        if 'design_df' not in st.session_state or st.session_state['design_df'] is None:
            st.error("Generate DOE first.")
        else:
            base = st.session_state['design_df'].copy().reset_index(drop=True)
            if 'D5' not in base.columns:
                base['D5'] = np.nan; base['D15'] = np.nan; base['D30'] = np.nan
            st.session_state['data_df'] = base
            st.success("Editable dataset created.")
    if 'data_df' in st.session_state:
        st.subheader("Current data (preview)")
        st.dataframe(st.session_state['data_df'].head(50))
        if st.button("Save dataset to session"):
            st.session_state['saved_at'] = datetime.datetime.now().isoformat()
            st.success("Dataset saved to session.")

# ---------------------------
# Modeling & analysis
# ---------------------------
st.markdown("---")
st.header("Modeling & Analysis")

if 'data_df' in st.session_state and st.session_state['data_df'] is not None:
    data_df = st.session_state['data_df'].copy().reset_index(drop=True)
    factor_cols = [c for c in factor_names if c in data_df.columns]
    response_cols = [c for c in data_df.columns if c not in factor_cols]
    st.write("Factors detected:", factor_cols)
    st.write("Responses detected:", response_cols)

    if len(response_cols) == 0:
        st.warning("No response columns detected.")
    else:
        # Coded RSM
        st.subheader("Coded-factor RSM")
        fit_coded = st.checkbox("Fit coded quadratic RSM (±1 coding) & show coefficients", value=True)
        if fit_coded:
            df_coded = build_coded_dataframe(data_df[factor_cols], bounds, factor_cols)
            df_full = pd.concat([data_df.reset_index(drop=True), df_coded], axis=1)
            df_terms = polynomial_terms_coded(df_full, factor_cols)
            st.write("Coded predictors (prefix c_) — preview:")
            display_cols = [f"c_{f}" for f in factor_cols] + [c for c in df_terms.columns if "__2" in c][:3]
            st.dataframe(df_terms[display_cols].head(6))
            coded_models = {}
            for r in response_cols:
                linear_terms = [f"c_{f}" for f in factor_cols]
                interaction_terms = []
                for i in range(len(factor_cols)):
                    for j in range(i+1, len(factor_cols)):
                        interaction_terms.append(f"c_{factor_names[i]}:c_{factor_names[j]}")
                square_terms = [f"I(c_{f}**2)" for f in factor_cols]
                formula = r + " ~ " + " + ".join(linear_terms + interaction_terms + square_terms)
                try:
                    model = smf.ols(formula, data=df_terms.assign(**{r: data_df[r].astype(float)})).fit()
                    coded_models[r] = model
                    st.subheader(f"Coded RSM for {r}")
                    coef_df = pd.DataFrame({
                        "term": model.params.index,
                        "coef": model.params.values,
                        "std_err": model.bse.values,
                        "pvalue": model.pvalues.values,
                        "t": model.tvalues
                    }).reset_index(drop=True)
                    st.dataframe(coef_df.style.format({"coef":"{:.4f}", "std_err":"{:.4f}", "pvalue":"{:.4f}", "t":"{:.3f}"}))
                except Exception as e:
                    st.error(f"Failed coded RSM for {r}: {e}")
            st.session_state['coded_models_obj'] = coded_models
            st.session_state['df_terms'] = df_terms

        # Bootstrap PI options
        st.markdown("### Bootstrap Prediction Intervals (for coded RSM predictions)")
        do_boot = st.checkbox("Enable bootstrap PIs", value=True)
        n_boot = st.number_input("Bootstrap resamples", min_value=100, max_value=5000, value=500, step=100)
        alpha = st.slider("Alpha (two-sided)", 0.01, 0.20, 0.05)

        if do_boot and 'coded_models_obj' in st.session_state:
            st.write("Predict at a new point (real units) — bootstrap PIs will be produced.")
            pred_point = []
            col_in = st.columns(len(factor_cols))
            for i, f in enumerate(factor_cols):
                with col_in[i]:
                    v = st.number_input(f"{f}", value=float(np.mean([bounds[i][0], bounds[i][1]])), key=f"pred_{f}")
                    pred_point.append(v)
            coded_vals = real_to_coded(pred_point, bounds)
            # build Xnew_df
            Xnew = {f"c_{factor_cols[i]}": coded_vals[i] for i in range(len(factor_cols))}
            for i in range(len(factor_cols)):
                for j in range(i+1, len(factor_cols)):
                    Xnew[f"c_{factor_cols[i]}_x_c_{factor_cols[j]}"] = coded_vals[i] * coded_vals[j]
            for i in range(len(factor_cols)):
                Xnew[f"c_{factor_cols[i]}__2"] = coded_vals[i]**2
            Xnew_df = pd.DataFrame([Xnew])
            pi_results = {}
            for r, mod in st.session_state['coded_models_obj'].items():
                # reconstruct formula
                linear_terms = [f"c_{f}" for f in factor_cols]
                interaction_terms = []
                for i in range(len(factor_cols)):
                    for j in range(i+1, len(factor_cols)):
                        interaction_terms.append(f"c_{factor_names[i]}:c_{factor_names[j]}")
                square_terms = [f"I(c_{f}**2)" for f in factor_cols]
                formula = r + " ~ " + " + ".join(linear_terms + interaction_terms + square_terms)
                df_for_boot = st.session_state['df_terms'].copy()
                df_for_boot[r] = data_df[r].astype(float)
                median, lower, upper, samples = bootstrap_prediction_intervals(formula, df_for_boot, Xnew_df, n_boot=int(n_boot), alpha=float(alpha))
                pi_results[r] = {"median": float(median), "lower": float(lower), "upper": float(upper)}
            st.write("Bootstrap prediction intervals (median, lower, upper):")
            st.json(pi_results)

        # ---------------------------
        # WEIBULL modeling tab
        # ---------------------------
        st.markdown("### Weibull modeling (multi-timepoint dissolution)")
        d_cols = [c for c in response_cols if c.lower().startswith('d')]
        if len(d_cols) >= 3:
            st.info(f"Detected timepoint columns: {d_cols}")
            if st.button("Fit Weibull to runs & model parameters"):
                times = [float(c[1:]) for c in d_cols]
                times_sorted_idx = np.argsort(times)
                times = np.array([times[i] for i in times_sorted_idx])
                ordered_cols = [d_cols[i] for i in times_sorted_idx]
                params_list = []
                for _, row in data_df.iterrows():
                    y = row[ordered_cols].astype(float).values
                    p = fit_weibull(times, y)
                    params_list.append(p)
                df_params = pd.DataFrame(params_list, columns=['A','k','b'])
                df_params = pd.concat([data_df[factor_cols].reset_index(drop=True), df_params], axis=1)
                st.session_state['weibull_params_df'] = df_params
                st.write("Weibull params (first rows):")
                st.dataframe(df_params.head(10))
                # fit coded RSM to A,k,b
                df_coded_params = build_coded_dataframe(df_params[factor_cols], bounds, factor_cols)
                df_coded_params = pd.concat([df_params.reset_index(drop=True), df_coded_params], axis=1)
                df_coded_params_terms = polynomial_terms_coded(df_coded_params, factor_cols)
                weibull_models = {}
                for pcol in ['A','k','b']:
                    formula = pcol + " ~ " + " + ".join([f"c_{f}" for f in factor_cols] +
                                                       [f"c_{factor_names[i]}:c_{factor_names[j]}" for i in range(len(factor_names)) for j in range(i+1,len(factor_names))] +
                                                       [f"I(c_{f}**2)" for f in factor_cols])
                    try:
                        mod = smf.ols(formula, data=df_coded_params_terms.assign(**{pcol: df_params[pcol].astype(float)})).fit()
                        weibull_models[pcol] = mod
                        st.write(f"Model for {pcol}:")
                        st.text(mod.summary())
                    except Exception as e:
                        st.error(f"Failed to model {pcol}: {e}")
                st.session_state['weibull_models'] = weibull_models
        else:
            st.info("Add at least three D* columns (e.g., D5,D15,D30) for Weibull modeling.")

        # ---------------------------
        # Contour & 3D surface plotting
        # ---------------------------
        st.markdown("---")
        st.subheader("Contour & 3D Surface Explorer (pick 2 factors)")
        if len(factor_cols) >= 2:
            f1 = st.selectbox("X factor", factor_cols, index=0, key="cont_f1")
            f2 = st.selectbox("Y factor", factor_cols, index=1, key="cont_f2")
            resp_for_plot = st.selectbox("Response to plot", response_cols, index=0)
            grid_n = st.slider("Grid resolution per axis", 20, 80, 40)
            bx = np.linspace(bounds[factor_names.index(f1)][0], bounds[factor_names.index(f1)][1], grid_n)
            by = np.linspace(bounds[factor_names.index(f2)][0], bounds[factor_names.index(f2)][1], grid_n)
            BX, BY = np.meshgrid(bx, by)
            Z = np.zeros_like(BX)
            # we need a predictor: prefer coded_models if response modeled
            for i in range(BX.shape[0]):
                for j in range(BX.shape[1]):
                    point = []
                    for k, fn in enumerate(factor_cols):
                        if fn == f1:
                            point.append(BX[i,j])
                        elif fn == f2:
                            point.append(BY[i,j])
                        else:
                            # use center of bound for other factors
                            point.append(np.mean(bounds[k]))
                    preds = {}
                    # predict single response
                    if 'coded_models_obj' in st.session_state and resp_for_plot in st.session_state['coded_models_obj']:
                        mod = st.session_state['coded_models_obj'][resp_for_plot]
                        coded_vals = real_to_coded(point, bounds)
                        dfp = {}
                        for idx, f in enumerate(factor_cols):
                            dfp[f"c_{f}"] = coded_vals[idx]
                        for ii in range(len(factor_cols)):
                            for jj in range(ii+1, len(factor_cols)):
                                dfp[f"c_{factor_cols[ii]}_x_c_{factor_cols[jj]}"] = coded_vals[ii]*coded_vals[jj]
                        for ii in range(len(factor_cols)):
                            dfp[f"c_{factor_cols[ii]}__2"] = coded_vals[ii]**2
                        try:
                            Z[i,j] = float(mod.predict(pd.DataFrame([dfp]))[0])
                        except Exception:
                            Z[i,j] = np.nan
                    else:
                        # fallback: nearest neighbor interpolation from data
                        # find nearest run in design/data and use measured resp if available
                        try:
                            # compute distances
                            pts = data_df[factor_cols].values
                            dists = np.linalg.norm(pts - np.array(point), axis=1)
                            idx = np.argmin(dists)
                            Z[i,j] = float(data_df.iloc[idx][resp_for_plot])
                        except Exception:
                            Z[i,j] = np.nan
            # Contour
            fig = go.Figure(data=go.Contour(z=Z, x=bx, y=by, colorscale="Viridis", contours=dict(showlabels=True)))
            fig.add_trace(go.Scatter(x=data_df[f1], y=data_df[f2], mode='markers', marker=dict(color='white', size=7, line=dict(color='black', width=1)), name='Runs'))
            fig.update_layout(title=f"Contour of {resp_for_plot} ({f1} vs {f2})", xaxis_title=f1, yaxis_title=f2)
            st.plotly_chart(fig, use_container_width=True)
            # 3D surface
            fig3 = go.Figure(data=[go.Surface(z=Z, x=bx, y=by)])
            fig3.update_layout(title=f"Surface: {resp_for_plot}", scene=dict(xaxis_title=f1, yaxis_title=f2, zaxis_title=resp_for_plot))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("At least 2 factors needed for contour/3D.")

        # ---------------------------
        # Optimization (with constraints)
        # ---------------------------
        st.markdown("---")
        st.subheader("Optimization (multi-response desirability)")
        include_constraints = st.checkbox("Enable constraints (linear/nonlinear)", value=True)
        linear_constraint_spec = {}
        if include_constraints:
            sel_factors_for_sum = st.multiselect("Pick factors to sum for linear constraint", factor_cols)
            if sel_factors_for_sum:
                limit_val = st.number_input("Upper limit for sum", value=float(sum([bounds[factor_names.index(f)][1] for f in sel_factors_for_sum])))
                linear_constraint_spec = {'factors': sel_factors_for_sum, 'limit': float(limit_val)}
        nonlinear_choice = st.selectbox("Nonlinear constraint", ["None", "product <= X", "ratio <= X"])
        nonlinear_spec = {}
        if nonlinear_choice != "None":
            f1n = st.selectbox("NL factor 1", factor_cols, key="nl1")
            f2n = st.selectbox("NL factor 2", factor_cols, key="nl2")
            nl_limit = st.number_input("NL limit", value=1.0, key="nl_limit")
            nonlinear_spec = {'type': nonlinear_choice, 'f1': f1n, 'f2': f2n, 'limit': float(nl_limit)}

        opt_resps = st.multiselect("Responses to include in optimization", response_cols, default=response_cols[:2])
        if len(opt_resps) > 0:
            desir_settings = {}
            st.markdown("Set desirability (per response)")
            for r in opt_resps:
                st.markdown(f"**{r}**")
                goal = st.selectbox(f"Goal for {r}", ["Maximize", "Minimize", "Target"], key=f"goal_opt_{r}")
                if goal == "Target":
                    tv = st.number_input(f"Target for {r}", value=float(data_df[r].mean()), key=f"tv_{r}")
                else:
                    L = st.number_input(f"{r} unacceptable L", value=float(data_df[r].min()), key=f"L_{r}")
                    T = st.number_input(f"{r} target T", value=float(data_df[r].max()), key=f"T_{r}")
                    tv = (float(L), float(T))
                w = st.slider(f"Weight for {r}", min_value=0.0, max_value=1.0, value=1.0/len(opt_resps), key=f"w_{r}")
                desir_settings[r] = {'goal': goal, 'target': tv, 'weight': w}
            # normalize weights
            total_w = sum([desir_settings[r]['weight'] for r in desir_settings])
            for r in desir_settings:
                desir_settings[r]['weight'] = desir_settings[r]['weight'] / max(total_w, 1e-9)

            def predict_point_real(x_list):
                preds = {}
                if 'coded_models_obj' in st.session_state:
                    for r in opt_resps:
                        if r in st.session_state['coded_models_obj']:
                            mod = st.session_state['coded_models_obj'][r]
                            coded_vals = real_to_coded(list(x_list), bounds)
                            dfp = {}
                            for i, f in enumerate(factor_cols):
                                dfp[f"c_{f}"] = coded_vals[i]
                            for i in range(len(factor_cols)):
                                for j in range(i+1, len(factor_cols)):
                                    dfp[f"c_{factor_cols[i]}_x_c_{factor_cols[j]}"] = coded_vals[i]*coded_vals[j]
                            for i in range(len(factor_cols)):
                                dfp[f"c_{factor_cols[i]}__2"] = coded_vals[i]**2
                            try:
                                preds[r] = float(mod.predict(pd.DataFrame([dfp]))[0])
                            except Exception:
                                preds[r] = float(np.nan)
                        else:
                            preds[r] = float(np.nan)
                else:
                    for r in opt_resps:
                        preds[r] = float(np.nan)
                return preds

            def composite_desirability_vector(xvec):
                preds = predict_point_real(xvec)
                prod = 1.0
                for r in desir_settings:
                    w = desir_settings[r]['weight']
                    g = desir_settings[r]['goal']
                    targ = desir_settings[r]['target']
                    p = preds.get(r, np.nan)
                    if np.isnan(p):
                        d = 0.0
                    else:
                        if g == "Maximize":
                            L, T = (targ if isinstance(targ, (list,tuple)) else (float(np.nanmin(data_df[r])), float(targ)))
                            d = desirability_maximize(p, L, T, s=1.0)
                        elif g == "Minimize":
                            L, T = (targ if isinstance(targ, (list,tuple)) else (float(targ), float(np.nanmax(data_df[r]))))
                            d = desirability_minimize(p, L, T, s=1.0)
                        else:
                            tv = targ
                            tol = 0.2*abs(tv) if tv!=0 else 1.0
                            d = 1.0 if (tv - tol) <= p <= (tv + tol) else 0.0
                    prod *= (d ** w)
                return prod

            # constraints
            constrs = []
            if linear_constraint_spec:
                idxs = [factor_cols.index(f) for f in linear_constraint_spec['factors']]
                A = np.zeros((1, len(factor_cols)))
                for i in idxs:
                    A[0,i] = 1.0
                ub = np.array([linear_constraint_spec['limit']])
                lc = LinearConstraint(A, -1e9, ub)
                constrs.append(lc)
            if nonlinear_spec:
                def nl_fun(x):
                    if nonlinear_spec['type'] == 'product <= X':
                        a = x[factor_cols.index(nonlinear_spec['f1'])]
                        b = x[factor_cols.index(nonlinear_spec['f2'])]
                        return nonlinear_spec['limit'] - (a*b)
                    elif nonlinear_spec['type'] == 'ratio <= X':
                        a = x[factor_cols.index(nonlinear_spec['f1'])]
                        b = x[factor_cols.index(nonlinear_spec['f2'])]
                        r = a / (b + 1e-9)
                        return nonlinear_spec['limit'] - r
                    else:
                        return 1.0
                nlcon = NonlinearConstraint(lambda x: nl_fun(x), 0.0, 1e9)
                constrs.append(nlcon)

            if st.button("Run constrained optimization"):
                grid_n = 9
                grids = [np.linspace(bounds[i][0], bounds[i][1], grid_n) for i in range(len(bounds))]
                mesh = np.meshgrid(*grids)
                flat = np.vstack([m.flatten() for m in mesh]).T
                best_val = -1; best_x = None
                for row in flat:
                    val = composite_desirability_vector(row)
                    if val > best_val:
                        best_val = val; best_x = row.copy()
                def neg_obj(x):
                    for i in range(len(x)):
                        if x[i] < bounds[i][0] or x[i] > bounds[i][1]:
                            return 1.0
                    return -composite_desirability_vector(x)
                res = minimize(neg_obj, best_x, bounds=bounds, constraints=constrs, method='SLSQP', options={'ftol':1e-9, 'maxiter':200})
                if not res.success:
                    st.warning(f"Optimizer warning: {res.message}")
                opt_x = res.x
                opt_val = -res.fun
                st.success(f"Optimization done — composite desirability = {opt_val:.4f}")
                sol = {factor_cols[i]: float(opt_x[i]) for i in range(len(factor_cols))}
                st.json(sol)
                preds = predict_point_real(opt_x)
                st.write("Predicted responses at optimum:")
                st.json(preds)
                st.session_state['opt_solution'] = {'x': sol, 'preds': preds, 'desir': float(opt_val)}
                # Weibull profile if available
                if 'weibull_models' in st.session_state:
                    coded_vals = real_to_coded(list(opt_x), bounds)
                    dfp = {}
                    for i, f in enumerate(factor_cols):
                        dfp[f"c_{f}"] = coded_vals[i]
                    for i in range(len(factor_cols)):
                        for j in range(i+1, len(factor_cols)):
                            dfp[f"c_{factor_cols[i]}_x_c_{factor_cols[j]}"] = coded_vals[i] * coded_vals[j]
                    for i in range(len(factor_cols)):
                        dfp[f"c_{factor_cols[i]}__2"] = coded_vals[i]**2
                    dfp_df = pd.DataFrame([dfp])
                    pA = st.session_state['weibull_models']['A'].predict(dfp_df)[0] if 'A' in st.session_state['weibull_models'] else np.nan
                    pk = st.session_state['weibull_models']['k'].predict(dfp_df)[0] if 'k' in st.session_state['weibull_models'] else np.nan
                    pb = st.session_state['weibull_models']['b'].predict(dfp_df)[0] if 'b' in st.session_state['weibull_models'] else np.nan
                    times = np.linspace(0, 60, 100)
                    pred_profile = weibull(times, pA, pk, pb)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=times, y=pred_profile, mode='lines', name='Predicted Weibull profile'))
                    fig.update_layout(title="Predicted dissolution profile (Weibull)", xaxis_title="Time (min)", yaxis_title="% dissolved")
                    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Save project area
# ---------------------------
st.markdown("---")
st.header("Save / Export Project")
if st.session_state.get('project_name', None) is None:
    pname = st.text_input("Project name to save", value=f"project_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    st.session_state['project_name'] = pname
else:
    pname = st.session_state['project_name']
st.write(f"Saving to project: **{pname}** (user: {st.session_state['username']})")
if st.button("Save project to disk (JSON)"):
    payload = {
        'factor_names': st.session_state.get('factor_names'),
        'bounds': st.session_state.get('bounds'),
        'design_df': st.session_state['design_df'].to_json() if st.session_state.get('design_df') is not None else None,
        'data_df': st.session_state['data_df'].to_json() if st.session_state.get('data_df') is not None else None,
        'saved_at': datetime.datetime.now().isoformat()
    }
    fname = save_project(st.session_state['username'], pname, payload)
    st.success(f"Project saved to {fname}")
    st.experimental_rerun()

st.markdown("Download current data (CSV):")
if 'data_df' in st.session_state and st.session_state['data_df'] is not None:
    csv_bytes = st.session_state['data_df'].to_csv(index=False).encode('utf-8')
    st.download_button("Download data CSV", data=csv_bytes, file_name=f"{pname}_data.csv", mime='text/csv')
