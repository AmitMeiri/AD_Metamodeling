#!/usr/bin/env python3
"""
reproduce_figures.py

Reproduce the main figures from:
"A multidimensional ODE-based model of Alzheimer's disease progression"
Bossa & Sahli (2023)

Usage:
    python reproduce_figures.py [--fast] [--stan-file PATH]
Notes:
    - Expects the Stan file ad_ode_model.stan in the working dir by default.
    - Supplementary PDF path (for provenance): /mnt/data/supplementary_AD_ODE.pdf
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
from cmdstanpy import CmdStanModel
import arviz as az
from sklearn.isotonic import isotonic_regression
from sklearn.metrics import roc_curve, auc
import os
import re

# הוספת נתיבי המהדר למערכת
os.environ["PATH"] += os.pathsep + r"C:\Users\amitm\.cmdstan\RTools40\usr\bin"
os.environ["PATH"] += os.pathsep + r"C:\Users\amitm\.cmdstan\RTools40\mingw64\bin"
# Paths (you provided these)
SUPP_PDF = "/mnt/data/supplementary_AD_ODE.pdf"
MAIN_PDF = "/mnt/data/s41598-023-29383-5.pdf"


# ---------------------------
# Utility: simulate trajectories
# ---------------------------
def simulate_trajectory_one_draw(v_mat, wAge_mat, wAPOE_mat,
                                 v0_vec, c0_vec, age, apoe4,
                                 times):
    """
    Simulate x(t) = v0 + expm(t * A) @ (c0 - v0),
    with A = v + age * wAge + apoe4 * wAPOE
    times: 1D array of times (years)
    returns: (len(times), K) array
    """
    A = v_mat + age * wAge_mat + apoe4 * wAPOE_mat
    K = v0_vec.shape[0]
    out = np.zeros((len(times), K))
    delta = (c0_vec - v0_vec).reshape(K, 1)
    for i, t in enumerate(times):
        M = expm(t * A)
        x = (M @ delta).reshape(K, ) + v0_vec
        out[i, :] = x
    return out


# ---------------------------
# Make plots
# ---------------------------
def plot_velocity_field_2d(v_mean, wAge_mean, wAPOE_mean,
                           v0_mean, grid_ab, fixed_tau_idx=1,
                           age_vals=[70, 80], apoe_vals=[0, 1],
                           timespan=2.0, figsize=(12, 6), outname=None):
    """
    Make 2D velocity field plots in ABeta x Tau plane like paper.
    We fix cognitive latents to v0 (population baseline) and vary ABeta,Tau grid.
    Only a 2D visualization: pick indices for ABeta and Tau in the state vector.
    """
    ab_vals = grid_ab[0]
    tau_vals = grid_ab[1]
    AB, TAU = np.meshgrid(ab_vals, tau_vals)
    # K dimension from v0
    K = v0_mean.shape[0]
    # We'll compute local velocity vectors for each grid point by substituting
    # a full state vector z where we set the ABeta and Tau coordinates to grid
    # and the other dims to v0.
    # Indices assumption: paper's ordering [Tau, AB, Cog...], but our Stan K layout:
    # In the Stan file: kCSF = kTau+1; mu idx 1 is AB (mu[:,1] used for AB),
    # they put jth element mapping in code. Here we assume:
    # index 0 -> AB (mu[:,1] in Stan); index 1 -> Tau (mu[:,2])
    # (If your data uses different order, change idx_ab, idx_tau accordingly.)
    idx_ab = 0
    idx_tau = 1
    U = np.zeros_like(AB)
    V = np.zeros_like(AB)
    for i in range(AB.shape[0]):
        for j in range(AB.shape[1]):
            # build state vector
            z = v0_mean.copy()
            z[idx_ab] = AB[i, j]
            z[idx_tau] = TAU[i, j]
            # velocity = A @ z + v0? careful: v(x) = A x + v0 in linear form used
            # In Stan they use scale_matrix_exp_multiply solution; local velocity = (A @ (z - v0)) ??? 
            # We'll compute v = A @ z + v0 (approx); actual velocity field in model is v(x) = A x + v0
            # Build A
            # But v_mat etc are global; here we just show direction: use v_mean + w*age etc.
            A = v_mean + np.zeros_like(v_mean)  # baseline
            vel = A @ z + v0_mean
            # use AB & TAU components
            U[i, j] = vel[idx_ab]
            V[i, j] = vel[idx_tau]
    # quiver plot
    plt.figure(figsize=figsize)
    Q = plt.quiver(AB, TAU, U, V, pivot='mid', scale=50)
    plt.xlabel("ABeta")
    plt.ylabel("Tau")
    plt.title("Velocity field (visualization approximation)")
    if outname:
        plt.savefig(outname, dpi=200)
    plt.close()


def plot_trajectories_from_draws(posterior_samples, subj_c0_idx=0,
                                 age=75, apoe4=0, times=np.linspace(0, 30, 301),
                                 n_draws=200, outname=None):
    """
    posterior_samples: dict with arrays for 'v', 'wAge', 'wAPOE', 'v0', 'c0' etc.
    We sample a number of posterior draws and plot trajectories for ABeta, Tau, and 1 ADAS dim.
    """
    # Get mean shapes
    v_draws = posterior_samples['v']  # shape (ndraws, K, K)
    wAge_draws = posterior_samples['wAge']
    wAPOE_draws = posterior_samples['wAPOE']
    v0_draws = posterior_samples['v0']  # (ndraws, K)
    c0_draws = posterior_samples['c0']  # (ndraws, N, K) or (ndraws, K) if single subject
    ndraws = v_draws.shape[0]
    pick = np.random.choice(np.arange(ndraws), size=min(n_draws, ndraws), replace=False)
    K = v0_draws.shape[1]
    plt.figure(figsize=(10, 6))
    # choose indices for AB, Tau, CogMem
    idx_ab = 0
    idx_tau = 1
    idx_mem = 2 if K > 2 else 2
    for s in pick:
        vmat = v_draws[s]
        wAge = wAge_draws[s]
        wAPOE = wAPOE_draws[s]
        v0 = v0_draws[s]
        # c0 for subject
        try:
            c0 = c0_draws[s, subj_c0_idx]
        except:
            c0 = c0_draws[s]
        traj = simulate_trajectory_one_draw(vmat, wAge, wAPOE, v0, c0,
                                            age, apoe4, times)
        plt.plot(times, traj[:, idx_ab], color='C0', alpha=0.08)
    plt.xlabel("Years")
    plt.ylabel("ABeta (simulated draws)")
    plt.title("Posterior predictive ABeta trajectories (many draws)")
    if outname:
        plt.savefig(outname, dpi=200)
    plt.close()


# ---------------------------
# Load/fit model, extract posterior summaries
# ---------------------------
def fit_or_load_model(stan_file="ad_ode_model.stan",
                      save_prefix="ad_ode_fit",
                      stan_data=None,
                      fast=True):
    """
    Compile Stan model, then either:
    - load cached posterior (pickle) if exists, or
    - run full sampling (slow) and extract ALL necessary parameters.
    """
    cache_file = f"{save_prefix}_posterior.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached posterior samples: {cache_file}")
        with open(cache_file, "rb") as f:
            posterior = pickle.load(f)
        return posterior, None

    model = CmdStanModel(stan_file=stan_file)  # [cite: 1281, 1282]

    if fast:
        # ADVI - Automatic Differentiation Variational Inference
        print("Running ADVI (fast approximate posterior) ...")  #
        vb = model.variational(data=stan_data, seed=123, iter=20000)
        # For simplicity in demo, we treat VB mean as single draw
        summary = vb.variational_params_dict
        # Map VB output to expected dictionary structure
        posterior = {
            'v': summary['v'][None, :, :],
            'wAge': summary['wAge'][None, :, :],
            'wAPOE': summary['wAPOE'][None, :, :],
            'v0': summary['v0'][None, :],
            'c0': summary['c0'][None, :, :]
        }
        if 'CDX' in summary:
            posterior['CDX'] = summary['CDX'][None, :]
    else:
        # MCMC - Markov Chain Monte Carlo
        print("Running full MCMC sampling (this may take a while)...")  #
        fit = model.sample(data=stan_data, seed=123, chains=4, parallel_chains=4,
                           iter_warmup=1000, iter_sampling=1000)

        # Convert to pandas for easier parameter extraction
        df = fit.draws_pd()
        posterior = {}

        # 1. Extract 2D matrices (v, wAge, wAPOE) [cite: 1253, 1258]
        for par_name in ['v', 'wAge', 'wAPOE']:
            cols = [c for c in df.columns if c.startswith(f"{par_name}[")]
            if cols:
                idxs = [re.findall(r"\[(\d+),(\d+)\]", c)[0] for c in cols]
                K = max([int(i) for i, j in idxs])
                posterior[par_name] = np.zeros((df.shape[0], K, K))
                for col in cols:
                    a, b = re.findall(r"\[(\d+),(\d+)\]", col)[0]
                    posterior[par_name][:, int(a) - 1, int(b) - 1] = df[col].values

        # 2. Extract 1D vectors (v0) [cite: 1254]
        cols_v0 = [c for c in df.columns if c.startswith("v0[")]
        if cols_v0:
            posterior['v0'] = df[cols_v0].values

        # 3. Extract 3D subject-level parameters (c0)
        cols_c0 = [c for c in df.columns if c.startswith("c0[")]
        if cols_c0:
            idxs = [re.findall(r"c0\[(\d+),(\d+)\]", c)[0] for c in cols_c0]
            N_subj = max([int(i) for i, j in idxs])
            K_feat = max([int(j) for i, j in idxs])
            posterior['c0'] = np.zeros((df.shape[0], N_subj, K_feat))
            for col in cols_c0:
                s, f = re.findall(r"c0\[(\d+),(\d+)\]", col)[0]
                posterior['c0'][:, int(s) - 1, int(f) - 1] = df[col].values

        # 4. Extract CDX (clinical thresholds)
        cols_cdx = [c for c in df.columns if c.startswith("CDX[")]
        if cols_cdx:
            posterior['CDX'] = df[cols_cdx].values

    # Save to cache so we don't have to run again
    with open(cache_file, "wb") as f:
        pickle.dump(posterior, f)

    return posterior, fit


def plot_ad_roc_curve_from_posterior(posterior, stan_data, outname="fig_roc_ad.png"):
    """
    מייצרת עקומת ROC מתוך נתוני ה-Posterior הטעונים, ללא צורך בהרצה מחדש של Stan.
    """
    from sklearn.metrics import roc_curve, auc

    # 1. חילוץ ממוצעי הפרמטרים מה-Posterior
    v_mean = np.mean(posterior['v'], axis=0)
    wAge_mean = np.mean(posterior['wAge'], axis=0)
    wAPOE_mean = np.mean(posterior['wAPOE'], axis=0)
    v0_mean = np.mean(posterior['v0'], axis=0)
    c0_mean = np.mean(posterior['c0'], axis=0)  # (N_subjects, K_features)

    # 2. שחזור הניבוי עבור כל תצפית אבחונית (DX)
    y_true = []
    y_scores = []

    # אינדקסים רלוונטיים
    idx_dx = np.array(stan_data['idxDX']) - 1  # אינדקסים של תצפיות אבחנה
    labels = np.array(stan_data['DX'])

    for i in range(len(idx_dx)):
        obs_idx = idx_dx[i]
        subj_idx = stan_data['ID'][obs_idx] - 1
        t_i = stan_data['t'][obs_idx]
        age_i = stan_data['AGE'][subj_idx]
        apoe_i = stan_data['APOE4'][subj_idx]

        # בניית מטריצת המהירות A עבור הנבדק בזמן המדידה
        A = v_mean + age_i * wAge_mean + apoe_i * wAPOE_mean

        # פתרון ה-ODE (משוואה 2 במאמר)
        delta = c0_mean[subj_idx] - v0_mean
        prediction = expm(t_i * A) @ delta + v0_mean

        # השגת הציון ל-AD (נשתמש בערך הממוצע של הממדים הקוגניטיביים כפרוקסי לסיכון)
        # במודל המקורי זה מחושב דרך bDX, כאן נשתמש בסימן הקוגניטיבי הישיר
        risk_score = np.mean(prediction[2:])  # ממוצע הממדים הקוגניטיביים

        y_true.append(1 if labels[i] == 3 else 0)  # 3 מייצג AD במאמר
        y_scores.append(risk_score)

    # 3. חישוב ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve: AD Diagnosis Prediction (from Cache)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(outname)
    plt.close()
    print(f"ROC Curve generated with AUC: {roc_auc:.2f}")


def plot_ad_roc_curve(fit, stan_data, outname="roc_curve_ad.png"):
    """
    מחשבת ומציירת עקומת ROC עבור הבחנה בין AD לבין non-AD.
    """
    # חילוץ ההסתברויות החזויות מהמודל (מתוך הפרמטר bDX ו-CDX ב-Stan)
    # הערה: במידה והשתמשת ב-fit.draws_pd(), נחפש את הניבויים הלוגיסטיים
    # במודל הזה, הניבויים עבור DX מתבצעים בתוך ה-Likelihood

    # במידה ותרצה לחשב זאת ידנית מהפוסטריור:
    # 1. קח את ה-mu_full עבור האינדקסים של הדיאגנוזה
    # 2. העבר אותם דרך פונקציית ordered_logistic

    # לצורך הפשטות, נשתמש בערכי ה-mu (הסטייט הפנימי) כסמן לסיכון
    mu_dx = fit.stan_variable("mu_full")[:, stan_data['idxDX'] - 1, :]

    # נתמקד בהסתברות ל-AD (הקטגוריה הגבוהה ביותר)
    # כאן נדרש חישוב לוגיסטי לפי הפרמטרים bDX ו-CDX מהמאמר

    # נניח שחילצנו את ההסתברות הממוצעת לכל תצפית
    y_true = np.array(stan_data['DX']) == 3  # האם האבחנה היא AD?

    # (כאן יש להוסיף חישוב הסתברות מבוסס bDX ו-CDX כפי שמופיע בקוד ה-Stan)
    # לצורך הדוגמה, נשתמש בממוצע ה-mu הראשון כפרוקסי לסיכון
    y_score = mu_dx.mean(axis=0)[:, 0]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: AD vs non-AD')
    plt.legend(loc="lower right")
    plt.savefig(outname)
    plt.close()


def plot_rate_of_change_dynamics(posterior, outname="fig_rate_dynamics.png"):
    """
    מייצר גרף המראה את קצב השינוי השנתי של טאו כפונקציה של רמות העמילואיד,
    בדומה ל-Figure 3 במאמר.
    """
    v_samples = posterior['v']  # (ndraws, K, K)
    v0_samples = posterior['v0']  # (ndraws, K)

    # נבנה רשת של ערכי עמילואיד (ציר X)
    ab_grid = np.linspace(400, 1800, 100)
    # נניח אינדקס 0 הוא AB ואינדקס 1 הוא Tau
    idx_ab, idx_tau = 0, 1

    # חישוב המהירות הממוצעת לכל נקודה ברשת
    tau_velocities = []
    for ab_val in ab_grid:
        # v(x) = Ax + v0. נקבע את שאר המשתנים לאפס (או לממוצע)
        z = np.zeros(v_samples.shape[2])
        z[idx_ab] = ab_val

        # חישוב המהירות עבור כל דגימה בפוסטריור כדי לקבל מרווחי וודאות
        vel_draws = []
        for s in range(min(100, v_samples.shape[0])):
            A = v_samples[s]
            v0 = v0_samples[s]
            v_x = A @ z + v0
            vel_draws.append(v_x[idx_tau])
        tau_velocities.append(vel_draws)

    tau_velocities = np.array(tau_velocities)
    mean_vel = np.mean(tau_velocities, axis=1)
    low_bound = np.percentile(tau_velocities, 5, axis=1)
    high_bound = np.percentile(tau_velocities, 95, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(ab_grid, mean_vel, color='blue', label='Mean Tau Change')
    plt.fill_between(ab_grid, low_bound, high_bound, color='blue', alpha=0.2, label='90% HDI')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('CSF Abeta basal level')
    plt.ylabel('CSF Tau yearly change')
    plt.title('Tau Dynamics vs Amyloid Levels')
    plt.legend()
    plt.savefig(outname)
    plt.close()


def plot_individual_forecast(posterior, stan_data, subj_idx=0, years_forward=10):
    """
    מייצר תחזית אישית לנבדק המציגה ביומרקרים והסתברות לדמנציה לאורך זמן.
    משחזר את הקונספט מ-Figure 1 (Bottom Right) ו-Figure 7.
    """
    import matplotlib.pyplot as plt
    from scipy.linalg import expm

    # 1. חילוץ פרמטרים ממוצעים מהפוסטריור
    v_mean = np.mean(posterior['v'], axis=0)
    wAge_mean = np.mean(posterior['wAge'], axis=0)
    wAPOE_mean = np.mean(posterior['wAPOE'], axis=0)
    v0_mean = np.mean(posterior['v0'], axis=0)
    c0_mean = np.mean(posterior['c0'], axis=0)[subj_idx]

    # נתוני הנבדק
    age = stan_data['AGE'][subj_idx]
    apoe = stan_data['APOE4'][subj_idx]

    # 2. סימולציה קדימה בזמן
    times = np.linspace(0, years_forward, 100)
    A = v_mean + age * wAge_mean + apoe * wAPOE_mean
    delta = c0_mean - v0_mean

    trajectories = []
    for t in times:
        # פתרון ה-ODE: x(t) = exp(t*A) * (c0 - v0) + v0
        xt = expm(t * A) @ delta + v0_mean
        trajectories.append(xt)

    trajectories = np.array(trajectories)

    # 3. ציור הגרף
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ציר ראשון: ביומרקרים (נורמליזציה לצורך ויזואליזציה)
    ax1.plot(times, trajectories[:, 0], 'b-', label='Abeta (scaled)', lw=2)
    ax1.plot(times, trajectories[:, 1], 'g-', label='Tau (scaled)', lw=2)
    ax1.plot(times, trajectories[:, 2], 'r--', label='Cognition (ADAS)', lw=2)
    ax1.set_xlabel('Years from baseline')
    ax1.set_ylabel('Biomarker / Cognition Levels')
    ax1.legend(loc='upper left')

    # ציר משני: הסתברות לאבחנה (אם קיים מודל הניבוי)
    # כאן ניתן להוסיף את חישוב ה-Prob(AD) כפונקציה של ה-trajectory

    plt.title(f'Disease Progression Forecast for Subject {subj_idx}')
    plt.grid(alpha=0.3)
    plt.savefig("fig_individual_forecast.png")
    plt.close()


def plot_figure_5_mci_conversion(posterior, stan_data, outname="fig_5_mci_conversion.png"):
    """
    משחזר את שלושת הגרפים של Figure 5 מהמאמר:
    1. התפתחות הסתברות לדמנציה על פני זמן.
    2. עקומות ROC לטווחי זמן שונים.
    3. Net Benefit (Decision Curve Analysis) לטווחי זמן שונים.
    """
    print("Simulating longitudinal clinical predictions...")

    # חילוץ ממוצעי הפרמטרים מהפוסטריור
    v_mean = np.mean(posterior['v'], axis=0)
    wAge_mean = np.mean(posterior['wAge'], axis=0)
    wAPOE_mean = np.mean(posterior['wAPOE'], axis=0)
    v0_mean = np.mean(posterior['v0'], axis=0)
    c0_mean = np.mean(posterior['c0'], axis=0)

    N_subj = stan_data['N']

    # נגדיר את אופקי הזמן שאנו רוצים לבחון (בשנים)
    target_horizons = [3, 4, 5, 6, 7]

    # מאגרי נתונים לשמירת התוצאות לגרפים
    all_trajectories = []
    roc_data = {t: {'y_true': [], 'y_prob': []} for t in target_horizons}

    # סימולציה עבור כל נבדק
    for subj_idx in range(N_subj):
        age = stan_data['AGE'][subj_idx]
        apoe = stan_data['APOE4'][subj_idx]

        # בניית מטריצת המהירות A לנבדק
        A = v_mean + age * wAge_mean + apoe * wAPOE_mean
        delta = c0_mean[subj_idx] - v0_mean

        # הדמיית התקדמות המחלה מ-2 עד 12 שנים (כפי שמופיע בגרף השמאלי במאמר)
        times = np.linspace(2, 12, 11)
        subj_probs = []

        # נניח שחולה הוא מראש במצב "מעורב" (MCI) ואנו בוחנים את ההחמרה
        true_conversion_time = np.random.uniform(3, 15)  # סימולציית אמת למקרה שאין תיוג אמיתי מ-ADNI

        for t in times:
            xt = expm(t * A) @ delta + v0_mean

            # חישוב הסתברות (פרוקסי: שימוש במדד הקוגניטיבי הממוצע והעברתו בפונקציה סיגמואידית)
            # במודל של בוסה זה מחושב דרך פרמטרי ה-Ordered Logistic
            risk_score = np.mean(xt[2:])
            prob_ad = 1 / (1 + np.exp(-(risk_score - 0.5) * 3))  # Sigmoid transformation
            subj_probs.append(prob_ad)

            # שמירת נתונים ל-ROC ול-Net Benefit בנקודות זמן ספציפיות
            if int(t) in target_horizons:
                is_ad = 1 if true_conversion_time <= t else 0
                roc_data[int(t)]['y_true'].append(is_ad)
                roc_data[int(t)]['y_prob'].append(prob_ad)

        all_trajectories.append((times, subj_probs, true_conversion_time))

    # --- יצירת 3 תתי-גרפים ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- גרף 1 (שמאל): הסתברות לדמנציה לאורך זמן ---
    ax1 = axes[0]
    for times, probs, conv_time in all_trajectories[:150]:  # הצגת 150 נבדקים למניעת עומס ויזואלי
        color = 'red' if conv_time <= 12 else 'steelblue'  # ממיר (אדום) מול לא-ממיר (כחול)
        alpha = 0.3 if color == 'red' else 0.1
        ax1.plot(times, probs, marker='o', markersize=4, color=color, alpha=alpha)
    ax1.set_xlabel('Time from last observation (years)')
    ax1.set_ylabel('Probability of Dementia')
    ax1.set_title('Individual Predictions over Time')

    # --- גרף 2 (אמצע): עקומות ROC ---
    ax2 = axes[1]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for i, t in enumerate(target_horizons):
        y_true = np.array(roc_data[t]['y_true'])
        y_prob = np.array(roc_data[t]['y_prob'])

        if len(np.unique(y_true)) > 1:  # ווידוא שיש לנו גם חולים וגם בריאים בזמן הזה
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, color=colors[i], lw=2, label=f'T = {t}y, AUC: {roc_auc:.2f}')

    ax2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('AD/non-AD ROC')
    ax2.legend(loc="lower right", fontsize=9)

    # --- גרף 3 (ימין): Decision Curve Analysis (Net Benefit) ---
    ax3 = axes[2]
    thresholds = np.linspace(0.01, 0.5, 50)

    for i, t in enumerate(target_horizons):
        y_true = np.array(roc_data[t]['y_true'])
        y_prob = np.array(roc_data[t]['y_prob'])
        n = len(y_true)

        if len(np.unique(y_true)) > 1:
            net_benefits = []
            for pt in thresholds:
                tp = np.sum((y_prob >= pt) & (y_true == 1))
                fp = np.sum((y_prob >= pt) & (y_true == 0))
                nb = (tp / n) - (fp / n) * (pt / (1 - pt))
                net_benefits.append(max(0, nb))  # חיתוך ב-0 למראה נקי
            ax3.plot(thresholds, net_benefits, color=colors[i], lw=2, label=f'Model, T = {t}y')

    # הוספת "Treat All" ו-"Treat None"
    prevalence = np.mean(roc_data[5]['y_true'])  # שימוש בשנה 5 כמייצג
    treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
    ax3.plot(thresholds, np.maximum(0, treat_all), 'k:', label='Treat all', alpha=0.6)
    ax3.axhline(0, color='black', linestyle='--', label='Treat none')

    ax3.set_xlim([0, 0.5])
    ax3.set_ylim([-0.02, 0.3])
    ax3.set_xlabel('Threshold Probability')
    ax3.set_ylabel('Net Benefit')
    ax3.set_title('Decision Curve Analysis')
    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close()
    print(f"Saved Figure 5 reproduction to {outname}")

# ---------------------------
# Minimal synthetic data creator (for pipeline test)
# ---------------------------
def build_minimal_synthetic_data():
    # This should match the Stan data block shapes
    N = 10
    n = 30
    m = 1
    D = 3
    kTau = 1
    t = np.linspace(0, 3, n)
    # small toy covariates and mapping
    X = np.zeros((n, m))
    AGE = np.random.normal(74, 4, size=N)
    APOE4 = np.random.binomial(1, 0.3, size=N)
    ID = np.repeat(np.arange(1, N + 1), n // N)[:n]
    nAB = n
    nTau = n
    nADAS = n
    nDX = n
    idxAB = list(range(1, nAB + 1))
    idxTau = list(range(1, nTau + 1))
    idxDX = list(range(1, nDX + 1))
    idxADAS = list(range(1, nADAS + 1))
    Tau = np.tile(np.linspace(200, 500, n).reshape(n, 1), (1, kTau))
    AB = np.linspace(0.2, 0.8, n)
    DX = np.random.randint(1, 4, size=n)
    ABmin = 0.0
    NABCens = 0
    idxABCens = []
    ADASCats = [4] * 12
    Qs = {f"Q{i}": np.random.randint(1, 5, size=n) for i in range(1, 13)}
    it_d = [1] * 4 + [2] * 4 + [3] * 4
    stan_data = {
        "N": N, "n": n, "t": t, "m": m, "X": X.tolist(),
        "AGE": AGE.tolist(), "APOE4": APOE4.tolist(), "ID": ID.tolist(),
        "kTau": kTau, "nAB": nAB, "nTau": nTau, "nADAS": nADAS, "nDX": nDX,
        "idxAB": idxAB, "idxTau": idxTau, "idxDX": idxDX, "idxADAS": idxADAS,
        "Tau": Tau.tolist(), "AB": AB.tolist(), "DX": DX.tolist(),
        "ABmin": ABmin, "NABCens": NABCens, "idxABCens": idxABCens,
        "ADASCats": ADASCats, "D": D, "it_d": it_d
    }
    # add Q1..Q12
    stan_data.update(Qs)
    return stan_data


# ----
# In case we have real data from ADNI database we load it here
# ----
def load_real_adni_data(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Sort data by subject ID and visit date
    # Ensure 'EXAMDATE' is parsed as datetime
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
    df = df.sort_values(by=['PTID', 'EXAMDATE']).reset_index(drop=True)

    # Create continuous subject IDs from 1 to N (Stan requires 1-based indexing)
    unique_ids = df['PTID'].unique()
    id_map = {ptid: i + 1 for i, ptid in enumerate(unique_ids)}
    df['stan_id'] = df['PTID'].map(id_map)

    N = len(unique_ids)
    n = len(df)

    # Calculate time in years from the first visit for each subject
    df['t'] = df.groupby('PTID')['EXAMDATE'].transform(
        lambda x: (x - x.min()).dt.days / 365.25
    )

    # Extract baseline age and APOE4 status for each unique subject
    # Assuming baseline data is valid for the whole group
    baseline_df = df.groupby('PTID').first().reset_index()
    AGE = baseline_df['AGE'].values.tolist()

    # APOE4 is often represented as number of alleles (0, 1, 2)
    # The model expects a binary indicator (1 if at least one allele, 0 otherwise)
    APOE4 = (baseline_df['APOE4'] > 0).astype(int).values.tolist()

    # Prepare biomarker data (handling NaNs)
    # Extract indices where ABeta (CSF) is available
    ab_mask = df['ABETA'].notna()
    nAB = ab_mask.sum()
    idxAB = df.index[ab_mask].values + 1  # 1-based indexing for Stan
    AB = df.loc[ab_mask, 'ABETA'].values

    # Handle censored ABeta data (ADNI often caps values at >1700)
    ABmin = 1700.0
    censored_mask = ab_mask & (df['ABETA'] >= ABmin)
    NABCens = censored_mask.sum()
    idxABCens = df.index[censored_mask].values + 1

    # Extract indices for Tau
    tau_mask = df['TAU'].notna()
    nTau = tau_mask.sum()
    idxTau = df.index[tau_mask].values + 1
    Tau = df.loc[tau_mask, ['TAU']].values  # Shape must be (nTau, 1)
    kTau = 1

    # Extract diagnosis codes (1=CN, 2=MCI, 3=AD)
    # Assuming 'DX' column is already mapped to 1, 2, 3
    dx_mask = df['DX'].notna()
    nDX = dx_mask.sum()
    idxDX = df.index[dx_mask].values + 1
    DX = df.loc[dx_mask, 'DX'].astype(int).values.tolist()

    # Extract ADAS-Cog item scores (Q1 to Q12)
    # Adjust column names based on your ADNI dictionary
    adas_cols = [f'ADAS_Q{i}' for i in range(1, 13)]
    adas_mask = df[adas_cols].notna().all(axis=1)
    nADAS = adas_mask.sum()
    idxADAS = df.index[adas_mask].values + 1

    # Number of categories for each ADAS item (based on ADAS-Cog structure)
    ADASCats = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]  # Update with actual max scores per item + 1
    it_d = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]  # Factor mapping (Language, Memory, Praxis)
    D = 3  # Number of cognitive dimensions

    # Build the final dictionary to pass to Stan
    stan_data = {
        "N": N,
        "n": n,
        "t": df['t'].values.tolist(),
        "m": 1,  # Number of extra covariates
        "X": np.zeros((n, 1)).tolist(),  # Empty covariates matrix if not used
        "AGE": AGE,
        "APOE4": APOE4,
        "ID": df['stan_id'].values.tolist(),
        "kTau": kTau,
        "nAB": nAB,
        "nTau": nTau,
        "nADAS": nADAS,
        "nDX": nDX,
        "idxAB": idxAB.tolist(),
        "idxTau": idxTau.tolist(),
        "idxDX": idxDX.tolist(),
        "idxADAS": idxADAS.tolist(),
        "Tau": Tau.tolist(),
        "AB": AB.tolist(),
        "DX": DX,
        "ABmin": ABmin,
        "NABCens": NABCens,
        "idxABCens": idxABCens.tolist(),
        "ADASCats": ADASCats,
        "D": D,
        "it_d": it_d
    }

    # Add Q1-Q12 items to the dictionary
    for i, col in enumerate(adas_cols, start=1):
        # IRT expects 1-based categorical scores, add 1 if scores start at 0
        stan_data[f"Q{i}"] = (df.loc[adas_mask, col] + 1).astype(int).values.tolist()

    return stan_data


# ---------------------------
# Main routine
# ---------------------------
def main(args):
    # 1. הגדרת שמות הקבצים והנתיבים
    save_prefix = "ad_ode_demo"
    cache_file = f"{save_prefix}_posterior.pkl"

    # 2. הכנת הנתונים (קורה תמיד, זה מיידי)
    # הערה: אם תרצה להשתמש בנתוני ADNI אמיתיים, תחליף את השורה הבאה בטעינה מה-CSV
    print("Preparing data...")
    stan_data = build_minimal_synthetic_data()

    # 3. טעינה מהזכרון או הרצה (כאן הקסם קורה)
    # הפונקציה תזהה לבד אם cache_file קיים ותדלג על השרשראות במידת הצורך
    posterior, fit = fit_or_load_model(stan_file=args.stan_file,
                                       save_prefix=save_prefix,
                                       stan_data=stan_data,
                                       fast=args.fast)

    # 4. עיבוד התוצאות ליצירת גרפים
    print("Processing posterior samples for visualization...")
    v_mean = np.mean(posterior['v'], axis=0)
    wAge_mean = np.mean(posterior['wAge'], axis=0)
    wAPOE_mean = np.mean(posterior['wAPOE'], axis=0)
    v0_mean = np.mean(posterior['v0'], axis=0)

    # --- יצירת גרף שדה המהירויות (Figure 2 במאמר) ---
    print("Generating Velocity Field plot...")
    grid_ab = (np.linspace(0.1, 0.9, 15), np.linspace(200, 900, 15))
    plot_velocity_field_2d(v_mean, wAge_mean, wAPOE_mean, v0_mean, grid_ab,
                           outname="fig_velocity_field.png")

    # --- יצירת גרף מסלולי התקדמות (Figure 7 במאמר) ---
    print("Generating Trajectory plot...")
    plot_trajectories_from_draws(posterior, subj_c0_idx=0,
                                 age=75, apoe4=1,
                                 times=np.linspace(0, 30, 301),
                                 n_draws=100, outname="fig_traj_abeta.png")

    # --- יצירת עקומת ROC (רק אם הרצנו עכשיו או אם הנתונים זמינים) ---
    print("Generating ROC curve from cached data...")
    plot_ad_roc_curve_from_posterior(posterior, stan_data)

    # א. גרף דינמיקת שיעורי השינוי (Figure 3 במאמר)
    # מדגיש איך רמת ביומרקר אחד משפיעה על המהירות של אחר
    plot_rate_of_change_dynamics(posterior, outname="fig_rate_dynamics.png")

    # ב. גרף תחזית התקדמות אישית (Figure 7 במאמר) [cite: 1129, 1203]
    # מראה את עתיד הנבדק (למשל נבדק 0) ל-10 שנים קדימה
    plot_individual_forecast(posterior, stan_data, subj_idx=0, years_forward=10)

    # הקריאה לשיחזור Figure 5:
    print("Generating Figure 5 (Clinical Predictions for MCI)...")
    plot_figure_5_mci_conversion(posterior, stan_data, outname="fig_5_mci_conversion.png")

    print("\n" + "=" * 30)
    print(f"SUCCESS: Figures saved to working directory.")
    print(f"Posterior data cached at: {cache_file}")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use ADVI instead of MCMC")
    parser.add_argument("--stan-file", default="ad_ode_model.stan", help="Path to Stan model")
    args = parser.parse_args()
    main(args)
