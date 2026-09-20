import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_geometry_plot():
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=200)

    beta1 = np.linspace(-2.5, 2.5, 500)
    beta2 = np.linspace(-2.5, 2.5, 500)
    B1, B2 = np.meshgrid(beta1, beta2)

    # Unconstrained OLS center
    ols_center = np.array([1.6, 1.4])
    # Elliptical loss contours (correlated features)
    loss = 1.5 * (B1 - ols_center[0])**2 + 2.5 * (B2 - ols_center[1])**2 - 1.8 * (B1 - ols_center[0]) * (B2 - ols_center[1])

    # Theme colors
    bg_color = '#0E1117'
    card_color = '#1A1C24'
    fig.patch.set_facecolor(bg_color)

    # -------------------------------------------------------------
    # 1. Ridge (L2) - Smooth Circle
    # -------------------------------------------------------------
    ax = axes[0]
    ax.set_facecolor(card_color)
    ax.axhline(0, color='#64748B', lw=1, ls='--', alpha=0.7)
    ax.axvline(0, color='#64748B', lw=1, ls='--', alpha=0.7)

    radius = 1.25
    ridge_constraint = B1**2 + B2**2
    ax.contourf(B1, B2, ridge_constraint, levels=[0, radius**2], colors=['#1E3A8A'], alpha=0.4)
    ax.contour(B1, B2, ridge_constraint, levels=[radius**2], colors=['#3B82F6'], linewidths=3.0)
    ax.contour(B1, B2, loss, levels=[0.2, 0.6, 1.2, 2.0, 3.2, 4.8], colors='#94A3B8', linewidths=1.3, linestyles=':')
    ax.plot(ols_center[0], ols_center[1], marker='x', color='#F87171', markersize=11, markeredgewidth=2.8, label=r'OLS odhad $\hat{\beta}$')
    ax.plot(0.88, 0.88, marker='o', color='#38BDF8', markersize=10, label=r'Optimum $\beta^*$ ($\beta_1 \neq 0, \beta_2 \neq 0$)')

    ax.set_title('Ridge (L₂)\nHladká kružnice (Žádné nulování)', fontsize=14, fontweight='bold', color='#60A5FA', pad=14)
    ax.set_xlabel(r'$\beta_1$', fontsize=13)
    ax.set_ylabel(r'$\beta_2$', fontsize=13)
    ax.set_xlim(-2.2, 2.5)
    ax.set_ylim(-2.2, 2.5)
    ax.legend(loc='lower left', fontsize=9.5, framealpha=0.75, facecolor='#0B0F19', edgecolor='#334155')
    ax.grid(True, alpha=0.15)

    # -------------------------------------------------------------
    # 2. Lasso (L1) - Sharp Diamond
    # -------------------------------------------------------------
    ax = axes[1]
    ax.set_facecolor(card_color)
    ax.axhline(0, color='#64748B', lw=1, ls='--', alpha=0.7)
    ax.axvline(0, color='#64748B', lw=1, ls='--', alpha=0.7)

    t_lasso = 1.25
    lasso_constraint = np.abs(B1) + np.abs(B2)
    ax.contourf(B1, B2, lasso_constraint, levels=[0, t_lasso], colors=['#78350F'], alpha=0.4)
    ax.contour(B1, B2, lasso_constraint, levels=[t_lasso], colors=['#F59E0B'], linewidths=3.0)
    ax.contour(B1, B2, loss, levels=[0.2, 0.6, 1.2, 2.0, 3.2, 4.8], colors='#94A3B8', linewidths=1.3, linestyles=':')
    ax.plot(ols_center[0], ols_center[1], marker='x', color='#F87171', markersize=11, markeredgewidth=2.8)
    ax.plot(0.0, 1.25, marker='o', color='#FBBF24', markersize=10, label=r'Optimum v rohu ($\beta_1 = 0$, Sparsity)')

    ax.set_title('Lasso (L₁)\nOstrý kosočtverec (Čisté nulování)', fontsize=14, fontweight='bold', color='#FBBF24', pad=14)
    ax.set_xlabel(r'$\beta_1$', fontsize=13)
    ax.set_ylabel(r'$\beta_2$', fontsize=13)
    ax.set_xlim(-2.2, 2.5)
    ax.set_ylim(-2.2, 2.5)
    ax.legend(loc='lower left', fontsize=9.5, framealpha=0.75, facecolor='#0B0F19', edgecolor='#334155')
    ax.grid(True, alpha=0.15)

    # -------------------------------------------------------------
    # 3. Elastic Net (L1 + L2) - Rounded Diamond
    # -------------------------------------------------------------
    ax = axes[2]
    ax.set_facecolor(card_color)
    ax.axhline(0, color='#64748B', lw=1, ls='--', alpha=0.7)
    ax.axvline(0, color='#64748B', lw=1, ls='--', alpha=0.7)

    rho = 0.65
    t_en = 1.15
    en_constraint = rho * (np.abs(B1) + np.abs(B2)) + (1 - rho) * (B1**2 + B2**2)
    ax.contourf(B1, B2, en_constraint, levels=[0, t_en], colors=['#064E3B'], alpha=0.45)
    ax.contour(B1, B2, en_constraint, levels=[t_en], colors=['#10B981'], linewidths=3.0)
    ax.contour(B1, B2, loss, levels=[0.2, 0.6, 1.2, 2.0, 3.2, 4.8], colors='#94A3B8', linewidths=1.3, linestyles=':')
    ax.plot(ols_center[0], ols_center[1], marker='x', color='#F87171', markersize=11, markeredgewidth=2.8)
    
    # Touch point at apex or on curved contour
    touch_y = (-rho + np.sqrt(rho**2 + 4 * (1 - rho) * t_en)) / (2 * (1 - rho))
    ax.plot(0.0, touch_y, marker='o', color='#34D399', markersize=10, label=r'Ostré rohy (Sparsity) + Oblouky (Grouping)')

    ax.set_title('Elastic Net (L₁ + L₂)\nZaoblený kosočtverec (Grouping effect)', fontsize=14, fontweight='bold', color='#34D399', pad=14)
    ax.set_xlabel(r'$\beta_1$', fontsize=13)
    ax.set_ylabel(r'$\beta_2$', fontsize=13)
    ax.set_xlim(-2.2, 2.5)
    ax.set_ylim(-2.2, 2.5)
    ax.legend(loc='lower left', fontsize=9.5, framealpha=0.75, facecolor='#0B0F19', edgecolor='#334155')
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    out_path = Path('01_Regression/plots/16_regularization_geometric_contours.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=bg_color)
    print(f'Uloženo: {out_path}')

if __name__ == '__main__':
    generate_geometry_plot()
