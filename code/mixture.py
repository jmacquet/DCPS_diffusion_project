import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HYPERPARAMÈTRES DCPS (Table 1 config) ---
dx = 10       # dimension of data
dy = 1          # dimension of observations
n_components = 25   # number of Gaussian components in the prior
L_blocks = 3       # L number of blocks
T_steps = 1000     # T = n number of diffusion steps
M_langevin = 50   # M number of Langevin steps per block
K_grad_steps = 2   # K number of SGD steps for mu, upsilon
zeta = 0.1         # Learning rate for optimizing mu, upsilon
gamma = 0.02     # Step size for Langevin
sigma_y = 0.05   # Noise on observations

k_intervals = np.linspace(0, T_steps, L_blocks + 1, dtype=int) # k_L = T, k_0 = 0
print("k_intervals:", k_intervals)

# --- INITIALISATION PRIOR GM ---
grid = torch.tensor([-2, -1, 0, 1, 2], device=device) * 8
grid_x, grid_y = torch.meshgrid(grid, grid, indexing='ij')
means_2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
means = torch.zeros((n_components, dx), device=device)
for k in range(0, dx, 2):
    means[:, k:k+2] = means_2d[:, :min(2, dx-k)]
weights = torch.distributions.Dirichlet(torch.ones(n_components)).sample().to(device)

# Noise Schedule
betas = torch.linspace(1e-4, 0.02, T_steps+1).to(device)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0) # alpha_t
print("Alphas_bar shape:", alphas_bar.shape)

# --- PROBLÈME INVERSE de sélections ---
"""
idx_a, idx_b = 3, 4
mu_a = means[idx_a]
mu_b = means[idx_b]
diff = (mu_a - mu_b).unsqueeze(1)
A = torch.randn(dy, dx, device=device)
A = A - (A @ diff) / (torch.norm(diff)**2) * diff.t()
y = A @ ((mu_a + mu_b).unsqueeze(1) / 2.0)
"""
A = torch.randn(dy, dx, device=device)
x_star = means[0] # On part d'un mode pour l'exemple
y = A @ x_star.unsqueeze(1) + torch.randn(dy, 1, device=device) * sigma_y #on part d'un y cohérent.

# --- FONCTIONS UTILES ---

def get_log_qt(xt, t_idx):
    ab = alphas_bar[t_idx]
    mu_t = torch.sqrt(ab) * means
    dists = -0.5 * torch.sum((xt.unsqueeze(0) - mu_t)**2, dim=1)
    return torch.logsumexp(torch.log(weights) + dists, dim=0)

def get_x0_hat(xt, t_idx):
    xt_c = xt.detach().requires_grad_(True)
    lqt = get_log_qt(xt_c, t_idx)
    score = torch.autograd.grad(lqt, xt_c)[0]
    ab = alphas_bar[t_idx]
    return (xt_c + (1 - ab) * score) / torch.sqrt(ab)

# --- COEFFICIENTS DU BRIDGE (A.2 & A.3) ---

def get_sigma_sq_lk(l_idx, k_idx):
    """Calcule sigma^2_{l|k} défini en (A.3) de manière stable."""
    # Sécurité pour les indices
    l_idx = int(max(0, l_idx))
    k_idx = int(max(0, k_idx))
    if l_idx >= k_idx:
        return torch.tensor(1e-10, device=device)
    ab_l = alphas_bar[l_idx]
    ab_k = alphas_bar[k_idx]
    
    # Formule A.3 avec epsilon de stabilité
    num = (1 - ab_l) * (1 - ab_k / ab_l)
    den = (1 - ab_k)
    res = num / (den + 1e-10)
    return res.clamp(min=1e-11) # Force un tenseur et une valeur positive

def get_mu_lk(l_idx, k_idx, xk):
    # On s'assure que les indices sont des entiers
    l_idx, k_idx = int(l_idx), int(k_idx)
    ab_l = alphas_bar[l_idx]
    ab_k = alphas_bar[k_idx]
    
    x0_h = get_x0_hat(xk, k_idx)
    
    # Coeffs A.2
    coeff_x0 = torch.sqrt(ab_l) * (1 - ab_k / ab_l) / (1 - ab_k + 1e-10)
    coeff_xk = torch.sqrt(ab_k / ab_l) * (1 - ab_l) / (1 - ab_k + 1e-10)
    
    return coeff_x0 * x0_h + coeff_xk * xk

# --- ESTIMATEUR BIAISÉ G_TILDE (A.9) ---

def get_log_gkl(x, l_idx):
    """
    Calcule log g_{k_l}(x) = log N(sqrt(alpha_bar_kl) * y; Ax, sigma_y^2 * I)
    Args:
        x: Le vecteur de taille dx (peut être x_perturbed dans G_tilde ou x_tilde dans Lj)
        l_idx: L'indice du bloc actuel (de L-1 à 0)
    """
    # 1. Récupération de l'indice de temps k_l associé au bloc l
    kl = k_intervals[l_idx] 
    ab_kl = alphas_bar[kl]
    sqrt_ab_kl = torch.sqrt(ab_kl)
    ax = A @ x.unsqueeze(1) # (dy, 1)
    target = sqrt_ab_kl * y.view(dy, 1)
    log_g_kl = -0.5 * torch.sum((target - ax)**2) / (sigma_y**2)
    return log_g_kl

def get_G_tilde_tamed(xkl1, l_idx, gamma):
    """
    Implémente G_tilde_gamma^ell défini en (A.9).
    k correspond ici au temps k_{l+1}.
    """
    # On travaille au niveau du bloc actuel (ell+1)
    kl = k_intervals[l_idx]
    kl1 = k_intervals[l_idx + 1]  # Dans Langevin on reste au même niveau temporel pour le potentiel
    x_c = xkl1.detach().requires_grad_(True)
    
    # 1. Calcul du score oracle (s_hat)
    lqt = get_log_qt(x_c, kl1) #calcul de log q_t 
    score_val = torch.autograd.grad(lqt, x_c)[0] #calcul de s_{l+1} = score = grad log q_t
    
    # 2. Estimation biaisée du gradient de log g (A.9)
    # On tire Z_l ~ N(0, I)
    z_l = torch.randn_like(x_c)
    mu_l = get_mu_lk(kl, kl1, x_c)
    sigma_l = torch.sqrt(get_sigma_sq_lk(kl, kl1))
    log_gl = get_log_gkl( mu_l + sigma_l * z_l, l_idx)
    grad_log_g = torch.autograd.grad(log_gl, x_c)[0]
    
    # 3. Assemblage et Taming (A.6)
    G_tilde = (grad_log_g + score_val) / (1+gamma*torch.norm(grad_log_g + score_val)) #ERREUR DANS LE PAPIER ?????
    
    return G_tilde

def get_Lj(mu_j_hat, v_j_hat, j_idx, l_idx, x):
    """
    Calcule la perte Lj définie en (A.12).
    Args:
        mu_j_hat: Le paramètre mu_j_hat optimisé
        v_j_hat: Le paramètre upsilon_j_hat optimisé
        j_idx: L'indice de temps j actuel
        l_idx: L'indice de bloc l actuel
        x_block: Le sample x_block courant (pour le double sampling)
    """
    # Double sampling (Z, Z')
    z_s = torch.randn(dx, device=device)
    z_prime = torch.randn(dx, device=device)
    
    # Échantillon xtilde = mu + exp(upsilon/2)*z
    x_tilde = get_mu_lk(k_intervals[l_idx], j_idx, mu_j_hat + torch.exp(v_j_hat / 2) * z_s) + get_sigma_sq_lk(k_intervals[l_idx], j_idx)**0.5 * z_prime
    log_gkl = get_log_gkl(x_tilde, l_idx=l_idx)
    kl_pj = torch.norm(mu_j_hat - get_mu_lk(j_idx, j_idx+1, x))**2 / (2 * get_sigma_sq_lk(j_idx, j_idx+1))
    kl_pj += -0.5 * torch.sum(v_j_hat - torch.exp(v_j_hat) / get_sigma_sq_lk(j_idx, j_idx+1), dim=0)
    return -log_gkl + kl_pj

# --- ALGORITHME DCPS COMPLET ---

def run_dcps(verbose=False):
    if verbose: print("Démarrage DCPS...")
    # Initial sample X_kL
    x_k = torch.randn(dx, device=device)
    
    # Boucle sur les blocs ell = L-1 à 0
    for ell in range(L_blocks - 1, -1, -1):
        k_next = k_intervals[ell]     # k_l
        k_curr = k_intervals[ell+1]   # k_{l+1}
        
        if verbose: print(f"Bloc {ell}: t={k_curr} -> t={k_next}")
        
        # --- ÉTAPE 1 : LANGEVIN (Guidage sur le bloc) ---
        x_block = x_k.clone()
        for i in range(M_langevin):
            g_val = get_G_tilde_tamed(x_k, ell, gamma)
            z = torch.randn(dx, device=device)
            x_block = x_block + gamma * g_val + torch.sqrt(torch.tensor(2 * gamma)) * z
        
        # --- ÉTAPE 2 : RECALAGE VARIATIONNEL & ÉCHANTILLONNAGE ---
        for j in range(k_curr - 1, k_next - 1, -1):
            # Initialisation de mu_j_hat et upsilon_j_hat
            mu_j_hat = get_mu_lk(j, j+1, x_block).detach().requires_grad_(True)
            v_j_hat = torch.log(get_sigma_sq_lk(j, j+1)).detach().requires_grad_(True)
            # Optimisation de mu et upsilon (K steps)
            for r in range(K_grad_steps):
                loss_y = get_Lj(mu_j_hat, v_j_hat, j, ell, x_block)
                grads = torch.autograd.grad(loss_y, [mu_j_hat, v_j_hat])
                for param, g in zip([mu_j_hat, v_j_hat], grads):
                    gnorm = torch.norm(g) + 1e-8
                    param.data -= zeta * (g / gnorm)
            # Échantillonnage final pour x_j
            eps = torch.randn(dx, device=device)
            x_block = mu_j_hat.detach() + torch.exp(v_j_hat.detach() / 2) * eps
        x_k = x_block.clone() # On passe au bloc suivant
        if verbose:
            print(x_k.size())
            print(x_k[:5]) # Affichage des 5 premières dimensions du sample courant
    return x_k.cpu().numpy()

def sample_exact_posterior(n_samples=1000):
    """
    Échantillonnage exact p(x|y) pour un mélange de Gaussiennes
    selon les formules du papier (A.10, A.11).
    """
    I_dx = torch.eye(dx, device=device)
    I_dy = torch.eye(dy, device=device)
    precision = I_dx + (1.0 / (sigma_y**2)) * (A.T @ A)
    sigma_post = torch.inverse(precision) # Cette matrice est partagée par tous les modes
    log_w_tilde = []
    m_tilde = []
    S = (sigma_y**2) * I_dy + A @ A.T
    # On utilise log_prob pour la stabilité numérique (évite sum of prob <= 0)
    dist_marginal = torch.distributions.MultivariateNormal(torch.zeros(dy, device=device), S)

    for i in range(n_components):
        mi = means[i]
        rhs = (1.0 / (sigma_y**2)) * (A.T @ y.view(-1, 1)).view(-1) + mi
        m_i_post = sigma_post @ rhs
        m_tilde.append(m_i_post)
        
        # Nouveau poids : w_tilde_i ∝ wi * N(y; A @ mi, S)
        y_mean = (A @ mi.unsqueeze(1)).view(-1)
        log_wi_y = dist_marginal.log_prob(y.view(-1) - y_mean)
        log_w_tilde.append(torch.log(weights[i] + 1e-15) + log_wi_y)
    
    # Normalisation stable des poids
    log_w_stack = torch.stack(log_w_tilde)
    w_updated = torch.nn.functional.softmax(log_w_stack, dim=0)
    m_tilde_stack = torch.stack(m_tilde)

    # Echantillonage des composantes
    indices = torch.multinomial(w_updated, n_samples, replacement=True)
    # Échantillonnage gaussien : x = m_tilde + L @ epsilon
    L = torch.linalg.cholesky(sigma_post + 1e-10 * I_dx)
    epsilon = torch.randn(n_samples, dx, device=device)
    
    samples = m_tilde_stack[indices] + (L @ epsilon.T).T
    
    return samples.cpu().numpy()

# --- Visualisation comparative ---

# --- EXECUTION ET VISUALISATION ---
n_true_samples = 1000
n_dcps_samples = 200
final_samples = [run_dcps(verbose=False) for _ in tqdm(range(n_dcps_samples))]
for generated_sample in final_samples:
    print("Generated sample (first 5 dims):", generated_sample[:5])
final_samples = np.array(final_samples)

exact_samples = sample_exact_posterior(n_true_samples)

plt.figure(figsize=(8,8))
plt.scatter(means_2d.cpu()[:, 0], means_2d.cpu()[:, 1], c='gray', alpha=0.1, label='Prior Modes')
plt.scatter(exact_samples[:, 0], exact_samples[:, 1], c='blue', s=10, alpha=0.5, label='Exact Posterior')
plt.scatter(final_samples[:,0], final_samples[:,1], c='red', s=10, alpha=0.5, label='DCPS Final Samples')
plt.title(f"DCPS Final (dx={dx}, L={L_blocks}, M={M_langevin})")
plt.axis('equal')
plt.show()