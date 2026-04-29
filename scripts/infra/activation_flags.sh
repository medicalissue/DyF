#!/usr/bin/env bash
# Map a normalization variant name → main.py CLI flags.
#
# Used by orchestrate.sh and dryrun.sh. Single source of truth so that
# adding a new variant (e.g. dyf-large-a-init) is one case-arm here, not
# an N-place edit across every launcher.
#
# Variants:
#   ln              baseline LayerNorm (no swap)
#   dyt             DynamicTanh (DyT paper)
#   dyf-silu        DyF, ψ = SiLU,  a=0 init  (≡ Swish-floor at init)
#   dyf-gelu        DyF, ψ = GELU,  a=0 init  (≡ GELU-floor at init)
#   dyf-hard        DyF, ψ = ReLU,  a=0 init  (≡ ReLU at init)
#   dyf-silu-aneg   ψ=SiLU, a=-1.0  (probe (C3): negative-tail floor)
#   dyf-silu-apos   ψ=SiLU, a=+0.5
#   dyf-gelu-aneg   ψ=GELU, a=-1.0
#   dyf-gelu-apos   ψ=GELU, a=+0.5
#
# Aliases: `dyf` ≡ `dyf-silu` (default kernel matches spec §2.2 Swish-floor).

activation_flags() {
    local act="$1"
    case "$act" in
        ln)
            printf '%s' "--dynamic_tanh false --dynamic_floor false"
            ;;
        dyt)
            printf '%s' "--dynamic_tanh true  --dynamic_floor false"
            ;;
        dyf|dyf-silu)
            printf '%s' "--dynamic_tanh false --dynamic_floor true --dyf_kernel silu --dyf_a_init 0.0"
            ;;
        dyf-gelu)
            printf '%s' "--dynamic_tanh false --dynamic_floor true --dyf_kernel gelu --dyf_a_init 0.0"
            ;;
        dyf-hard)
            printf '%s' "--dynamic_tanh false --dynamic_floor true --dyf_kernel hard --dyf_a_init 0.0"
            ;;
        dyf-silu-aneg)
            printf '%s' "--dynamic_tanh false --dynamic_floor true --dyf_kernel silu --dyf_a_init -1.0"
            ;;
        dyf-silu-apos)
            printf '%s' "--dynamic_tanh false --dynamic_floor true --dyf_kernel silu --dyf_a_init 0.5"
            ;;
        dyf-gelu-aneg)
            printf '%s' "--dynamic_tanh false --dynamic_floor true --dyf_kernel gelu --dyf_a_init -1.0"
            ;;
        dyf-gelu-apos)
            printf '%s' "--dynamic_tanh false --dynamic_floor true --dyf_kernel gelu --dyf_a_init 0.5"
            ;;
        # ── Dynamic Saturation (DyS) ─────────────────────────────────
        # f(x) = γ · (x/a) · K · σ'(x/a) + β   where K normalises so
        # f'(0) = 1 (K=4 for dsilu, K=√(2π) for dgelu). a is per-site
        # input scale; sqrt schedule sets a_l = sqrt(l+1) to track
        # pre-LN residual stream std growth.
        # Peak-slope matching to DyT's α=0.5 init (DyT works with this
        # for ViT-B). The zero-slope of DyS = γ·g'(0)/a; setting it to
        # 0.5 with a=1 (DyT-like constant) gives:
        #   dgelu: g'(0) = 1/√(2π) = 0.399  →  γ = 0.5/0.399 = 1.253
        #   dsilu: g'(0) = 0.25            →  γ = 0.5/0.25  = 2.000
        # Output std then comes out layer-dependent (~0.21 dgelu,
        # ~0.31 dsilu) similar to DyT's own (~0.42-0.84) — not 1.
        # Self-consistency we earlier obsessed about isn't required
        # (DyT itself doesn't satisfy it).
        dys-dgelu)
            printf '%s' "--dynamic_saturation true --dys_kernel dgelu --dys_a_init 1.0 --dys_a_init_schedule constant --dys_gamma_init 1.253"
            ;;
        dys-dsilu)
            printf '%s' "--dynamic_saturation true --dys_kernel dsilu --dys_a_init 1.0 --dys_a_init_schedule constant --dys_gamma_init 2.000"
            ;;
        *)
            echo "ERROR: unknown activation '$act'." \
                "Valid: ln dyt dyf dyf-silu dyf-gelu dyf-hard" \
                "dyf-silu-{aneg,apos} dyf-gelu-{aneg,apos}" \
                "dys-dgelu dys-dsilu" >&2
            return 2
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    activation_flags "$@"
fi
