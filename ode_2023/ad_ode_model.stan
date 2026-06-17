data {
  int N;                          // Number of subjects
  int n;                          // Number of observations
  array[n] real t;                // Observation times
  int m;                          // Number of covariates
  matrix[n, m] X;                 // Covariates matrix
  array[N] real AGE;              // Subjects' age
  array[N] int<lower=0, upper=1> APOE4; // One APOE4 allele minimum
  array[n] int<lower=1, upper=N> ID;    // Subject ID for each observation

  int kTau;                       // Number of elements of Tau vector
  int nAB;                        // Number of valid observations for AB
  int nTau;                       // Number of valid observations for Tau
  int nADAS;                      // Number of valid observations for ADAS
  int nDX;                        // Number of valid observations for Diagnosis

  // Indexes to locate time (t) or ID for each measurement
  array[nAB] int<lower=1, upper=n> idxAB;
  array[nTau] int<lower=1, upper=n> idxTau;
  array[nDX] int<lower=1, upper=n> idxDX;
  array[nADAS] int<lower=1, upper=n> idxADAS;

  // Observations
  matrix[nTau, kTau] Tau;
  vector[nAB] AB;
  array[nDX] int<lower=1, upper=3> DX;

  // ABETA censored info
  real<lower=0, upper=1> ABmin;
  int<lower=0, upper=n> NABCens;
  array[NABCens] int<lower=1, upper=n> idxABCens;

  // ADAS
  array[12] int<lower=2> ADASCats; // Number of categories per item

  // Categorical data for ADAS items (Q1-Q12)
  array[nADAS] int<lower=1> Q1;
  array[nADAS] int<lower=1> Q2;
  array[nADAS] int<lower=1> Q3;
  array[nADAS] int<lower=1> Q4;
  array[nADAS] int<lower=1> Q5;
  array[nADAS] int<lower=1> Q6;
  array[nADAS] int<lower=1> Q7;
  array[nADAS] int<lower=1> Q8;
  array[nADAS] int<lower=1> Q9;
  array[nADAS] int<lower=1> Q10;
  array[nADAS] int<lower=1> Q11;
  array[nADAS] int<lower=1> Q12;

  int<lower=1> D;                // Number of dimensions/factors
  array[12] int<lower=1, upper=D> it_d; // Which factor corresponds to each item
}

transformed data {
  int kCSF = kTau + 1;
  int K = kCSF + D;            // Total number of features
  matrix[kCSF, D] zCSF = rep_matrix(0, kCSF, D); // Auxiliary variable with zeros
}

parameters {
  array[kCSF] vector[N] COCSF;       // Random effects (CSF) initial values

  array[kCSF] real<lower=0> sRECSF;  // Variance of CSF Random Effects
  array[kCSF] real muRECSF;          // Mean CSF Random Effects

  // Velocity field parameters
  matrix[kCSF, kCSF] VCSF;
  matrix[D, K] VADAS;
  vector[K] v0;

  matrix[kCSF, kCSF] WCSFAge;
  matrix[kCSF, kCSF] WCSFAPOE;
  matrix[D, K] WADASAge;
  matrix[D, K] WADASAPOE;

  // Observation noise variances
  array[kTau] real<lower=0> sTau;
  real<lower=0> sAB;

  // Prediction model parameters
  vector[K + m] bDX;
  ordered[2] CDX;

  // ADAS-Cog IRT model parameters
  vector<lower=0, upper=5>[12] alpha;

  // Thresholds for each item
  ordered[ADASCats[1]-1] th1;
  ordered[ADASCats[2]-1] th2;
  ordered[ADASCats[3]-1] th3;
  ordered[ADASCats[4]-1] th4;
  ordered[ADASCats[5]-1] th5;
  ordered[ADASCats[6]-1] th6;
  ordered[ADASCats[7]-1] th7;
  ordered[ADASCats[8]-1] th8;
  ordered[ADASCats[9]-1] th9;
  ordered[ADASCats[10]-1] th10;
  ordered[ADASCats[11]-1] th11;
  ordered[ADASCats[12]-1] th12;

  array[N] row_vector[D] COADAS;     // Factor scores

  real<lower=0, upper=5> sigma_alpha;
  cholesky_factor_corr[D] L_corr_d;
}

transformed parameters {
  // Velocity field parameters
  matrix[K, K] v;
  matrix[K, K] wAge;
  matrix[K, K] wAPOE;
  array[N] row_vector[K] c0;

  // Constructing the full matrices from blocks
  v[1:kCSF, 1:kCSF] = VCSF;
  v[1:kCSF, (kCSF + 1):K] = zCSF;
  v[(kCSF + 1):K, :] = VADAS;

  wAge[1:kCSF, 1:kCSF] = WCSFAge;
  wAge[1:kCSF, (kCSF + 1):K] = zCSF;
  wAge[(kCSF + 1):K, :] = WADASAge;

  wAPOE[1:kCSF, 1:kCSF] = WCSFAPOE;
  wAPOE[1:kCSF, (kCSF + 1):K] = zCSF;
  wAPOE[(kCSF + 1):K, :] = WADASAPOE;

  for (k in 1:kCSF) {
    c0[:, k] = to_array_1d(muRECSF[k] + to_vector(COCSF[k]) * sRECSF[k]);
  }
  c0[:, (1+kCSF):K] = COADAS;
}

model {
  matrix[n, K] mu_traj; // Trajectory values
  matrix[n, K + m] mu_full;

  // --- Priors ---
  for (k in 1:kTau) sTau[k] ~ normal(0, 0.2);
  sAB ~ normal(0, 0.2);

  bDX ~ normal(0, 100);

  for (k2 in 1:kCSF) {
    for (k1 in 1:kCSF) {
      WCSFAge[k1, k2] ~ normal(0, 0.02);
      WCSFAPOE[k1, k2] ~ normal(0, 0.02);
      VCSF[k1, k2] ~ normal(0, 0.1);
    }
  }

  v0 ~ normal(0, 0.1);

  for (k in 1:K) {
    for (d in 1:D) {
      WADASAge[d, k] ~ normal(0, 0.05);
      WADASAPOE[d, k] ~ normal(0, 0.05);
      VADAS[d, k] ~ normal(0, 0.1);
    }
  }

  sRECSF ~ std_normal();
  muRECSF ~ std_normal();

  for (k in 1:kCSF) COCSF[k] ~ std_normal();

  sigma_alpha ~ std_normal();
  alpha ~ lognormal(0, sigma_alpha);

  L_corr_d ~ lkj_corr_cholesky(1);
  COADAS ~ multi_normal_cholesky(rep_vector(0, D), L_corr_d);

  // --- Trajectory Calculation (ODE Solution) ---
  for (i in 1:n) {
    matrix[K, K] A_i = v + AGE[ID[i]] * wAge + APOE4[ID[i]] * wAPOE;
    matrix[K, 1] diff = to_matrix(to_vector(c0[ID[i]]) - v0);
    matrix[K, 1] dyn_mat = matrix_exp_multiply(A_i * t[i], diff);
    mu_traj[i] = to_row_vector(to_vector(dyn_mat) + v0);
  }

  mu_full[1:n, 1:K] = mu_traj;
  mu_full[1:n, (K+1):(K+m)] = X;

  // --- Likelihood ---
  AB ~ normal(mu_traj[idxAB, 1], sAB);

  for (i in 1:NABCens) {
      target += normal_lcdf(ABmin | mu_traj[idxABCens[i], 1], sAB);
  }

  for (k in 1:kTau) {
      Tau[:, k] ~ normal(mu_traj[idxTau, k + 1], sTau[k]);
  }

  {
      matrix[nDX, K+m] mu_dx_subset;
      for(i in 1:nDX) mu_dx_subset[i] = mu_full[idxDX[i]];
      DX ~ ordered_logistic_glm(mu_dx_subset, bDX, CDX);
  }

  // FIXED: Multiplied mu_traj by alpha inside the function call
  Q1 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[1]] * alpha[1], th1);
  Q2 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[2]] * alpha[2], th2);
  Q3 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[3]] * alpha[3], th3);
  Q4 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[4]] * alpha[4], th4);
  Q5 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[5]] * alpha[5], th5);
  Q6 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[6]] * alpha[6], th6);
  Q7 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[7]] * alpha[7], th7);
  Q8 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[8]] * alpha[8], th8);
  Q9 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[9]] * alpha[9], th9);
  Q10 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[10]] * alpha[10], th10);
  Q11 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[11]] * alpha[11], th11);
  Q12 ~ ordered_logistic(mu_traj[idxADAS, kCSF + it_d[12]] * alpha[12], th12);
}