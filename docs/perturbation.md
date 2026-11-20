# 2.1 Perturbation analysis: derivation skeleton

*A skeleton to write up in full: each step names what to show, the key equation, and the assumption it needs. Notation follows the code: n = 19 state factors, X_PH and Y_PH are n × N_PH.*

## 0. Setup and assumptions
- **Global model:** M_G x ≈ y, fitted on the international cells (Phase 1).
- **Philippine model:** M_PH x ≈ y on the PH cells, with **M_PH = M_G + ΔM**.
- **PH data as a perturbation of global data:** x_PH = x_G + Δx, y_PH = y_G + Δy. Here x_G and y_G are the global vectors of the *same* group (for example the global median Q1_RURAL cell).
- **Assumptions to state:**
  - A1. ΔM is small relative to M_G (‖ΔM‖ ≪ ‖M_G‖), so second-order terms can be dropped.
  - A2. The same n-dimensional state space and the same scaling (one global scaler) apply in both.
  - A3. Errors in the representative vectors are sampling noise of known size (the CLT standard errors from Phase 0).
  - A4. Linearity, and one matrix per cycle step (no intercept, unless one is added in Phase 1).

## (a) First-order relation and the minimum-norm solution
1. Expand (M_G + ΔM)(x_G + Δx) = y_G + Δy, then use M_G x_G = y_G:
   - M_G Δx + ΔM x_G + ΔM Δx = Δy
2. Drop the second-order term ΔM Δx (A1). *Show when this is justified: compare the size of ‖ΔM Δx‖ with that of the other terms.*
   - **ΔM x ≈ Δy − M_G Δx**
3. For one column there are n equations and n² unknowns, so infinitely many solutions. The minimum-Frobenius-norm one is
   - ΔM = r xᵀ / ‖x‖², with r = Δy − M_G Δx
   - *To show:* derive it with Lagrange multipliers, or note it is r x⁺.
4. For all PH columns stacked, the residual is R = Y_PH − M_G X_PH (this equals the stacked Δy − M_G Δx), and the system is
   - **ΔM X_PH = R**
   - The minimum-norm solution is **ΔM = R X_PH⁺** (via the SVD of X_PH).
   - *To show:* this is the exact fit when N_PH ≤ n and X_PH has full column rank.
5. *Discuss:* an exact fit to noisy PH data overfits, which motivates (b).

## (b) The stabilised problem
- If regularisation is used, the problem becomes min_M ‖M X_PH − Y_PH‖_F² + λ ‖M − M_G‖_F².
  - Substitute M = M_G + ΔM. It becomes min ‖ΔM X − R‖² + λ ‖ΔM‖², which is Tikhonov on the residual system.
  - Normal equations: ΔM (X Xᵀ + λI) = R Xᵀ, so **ΔM = R Xᵀ (X Xᵀ + λI)⁻¹**.
  - In SVD form, with X = U S Vᵀ: ΔM = R V diag(sᵢ / (sᵢ² + λ)) Uᵀ, so the filter factors damp the directions with small sᵢ.
  - Limits: λ → 0 gives (a), and λ → ∞ gives ΔM → 0 (M_PH = M_G).
- **Alternatives without λ** (see imple/perturbation.py): truncated SVD (keep k directions), low-rank ΔM, or ΔM restricted to chosen columns. *Pick one and justify it.*
- *Bayesian reading (optional):* a prior ΔM ~ N(0, τ²) with noise σ² gives λ = σ²/τ².

## (c) Error bounds
1. **Sensitivity of the solve.** For ΔM X = R with noise E_R in R and E_X in X, the relative error in ΔM is bounded to first order by
   - κ(X) (‖E_X‖/‖X‖ + ‖E_R‖/‖R‖)
   - with κ(X) = s_max / s_min from imple.svd.
   - *To show:* derive it from the pseudoinverse perturbation bound. *Interpret:* PH cells that are near-collinear make κ large and ΔM unreliable.
2. **Normal equations versus QR/SVD.** κ(X Xᵀ) = κ(X)². Reference the Phase 1.2 conditioning experiment.
3. **Weyl's inequality** for the spectra:
   - |σᵢ(M_PH) − σᵢ(M_G)| ≤ ‖ΔM‖₂ for every i.
   - *Use:* if ‖ΔM‖₂ is small, the conditioning and dominant directions of M_PH stay close to those of M_G.
4. **Eigenvalues (bridge to Phase 4):** Bauer–Fike gives |λ(M_PH) − λ(M_G)| ≤ κ(V) ‖ΔM‖₂, where V holds the eigenvectors of M_G (if M_G is diagonalisable).
   - *Note:* a non-normal M_G can have a large κ(V), so the eigenvalues can move much more than the singular values.
5. **Noise floor.** Propagate the Phase 0 standard errors: ‖E_R‖ ≈ ‖Y_se‖ + ‖M_G X_se‖. A ΔM entry is meaningful only where it exceeds the bound in (c1).

## (d) What to report
- ‖ΔM‖_F / ‖M_G‖_F and ‖ΔM‖₂ (A1 check), κ(X_PH), the bound from (c1) against the observed residual, and the Weyl check on the computed matrices.
- The assumptions that may fail:
  - A1 when the PH system is structurally different;
  - A4 because 2018 → 2022 includes COVID-19, and the long PH school closures act as a shock rather than a perturbation.
