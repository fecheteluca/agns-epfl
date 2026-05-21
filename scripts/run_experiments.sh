#!/usr/bin/env bash
#
# scripts/run_experiments.sh -- regenerate every benchmark artefact end
# to end.
#
# Pipeline per campaign:
#   1. python -m agns.cli.run_benchmark --config <cfg> --seeds <list>
#        -> results/numerical/raw/<campaign>/seed_*/<method>_history.pkl
#   2. python -m agns.cli.aggregate    --campaign <name> ...
#        -> results/numerical/aggregated/<campaign>.json
#   3. python -m agns.cli.make_figures --aggregated <json> ...
#        -> results/figures/<campaign>/*.{pdf,png}
#   4. python -m agns.cli.make_tables  --aggregated <json> ...
#        -> results/tables/<campaign>.tex
#
# After every real_* campaign has been aggregated, a cross-dataset
# rollup table is rendered:
#   results/tables/real_world_summary.tex
#
# Environment variables (all optional):
#   AGNS_SEEDS      Space-separated seed list (default: "0 1 2 3 4 5 6 7 8 9").
#                   For a quick smoke pass: AGNS_SEEDS="0 1 2".
#                   For a sanity run:       AGNS_SEEDS="0".
#   AGNS_RESULTS    Output root (default: ./results).
#   AGNS_SKIP_DATA  If non-empty, skip scripts/download_data.sh.
#   AGNS_SKIP_RUN   If non-empty, skip the benchmark + aggregation steps
#                   and only re-render figures and tables from the
#                   existing results/numerical/aggregated/ JSONs.
#   AGNS_DRY_RUN    If non-empty, print commands without executing them.
#   AGNS_PYTHON     Python interpreter (default: python).

set -euo pipefail
IFS=$'\n\t'

# Pin BLAS / OpenMP threads to 1 BEFORE any Python process starts.
# Multi-threaded BLAS reductions order their additions non-determin-
# istically, which drifts numerical fields by ~1 ULP between re-runs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AGNS_PYTHON="${AGNS_PYTHON:-python}"
AGNS_SEEDS="${AGNS_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
AGNS_RESULTS="${AGNS_RESULTS:-${REPO_ROOT}/results}"
AGNS_SKIP_DATA="${AGNS_SKIP_DATA:-}"
AGNS_SKIP_RUN="${AGNS_SKIP_RUN:-}"
AGNS_DRY_RUN="${AGNS_DRY_RUN:-}"

NUMERICAL_RAW="${AGNS_RESULTS}/numerical/raw"
NUMERICAL_AGG="${AGNS_RESULTS}/numerical/aggregated"
FIGURES_DIR="${AGNS_RESULTS}/figures"
TABLES_DIR="${AGNS_RESULTS}/tables"

# Create the output tree only for a real run; a dry run prints commands
# and must not touch the filesystem.
if [[ -z "${AGNS_DRY_RUN}" ]]; then
    mkdir -p "${NUMERICAL_RAW}" "${NUMERICAL_AGG}" "${FIGURES_DIR}" "${TABLES_DIR}"
fi

log() { printf '[run_experiments] %s\n' "$*" >&2; }
run() {
    if [[ -n "${AGNS_DRY_RUN}" ]]; then
        printf '[dry-run] %s\n' "$*"
    else
        eval "$@"
    fi
}

# ---------------------------------------------------------------------------
# Campaign list.  Format:  <config>  <campaign_name>  [<nu_ref>]
#
# ``nu_ref`` is optional; when present, make_figures.py overlays a
# reference slope ``C / K^{2+nu}`` on the rate_loglog figure.
#
# Add a row to publish a new sub-campaign.  Empty rows / # comments are
# skipped.
# ---------------------------------------------------------------------------

read -r -d '' CAMPAIGNS <<'EOF' || true
configs/rate_verification/logsumexp.yaml          rate_verification
configs/rate_verification/nle_p4.yaml             rate_verification_nle           1.0
configs/rate_verification/nle_p225.yaml           rate_verification_nle_nu025     0.25
configs/rate_verification/nle_p25.yaml            rate_verification_nle_nu050     0.5
configs/rate_verification/nle_p275.yaml           rate_verification_nle_nu075     0.75
configs/convex_lipschitz/logistic_synthetic.yaml  convex_lipschitz_lr
configs/convex_lipschitz/softmax_regression.yaml  softmax_regression
configs/holder_hessian/smoothed_lasso.yaml        holder_smoothed_lasso           0.5
configs/strongly_convex/ridge_synthetic.yaml      strongly_convex_ridge
configs/strongly_convex/logistic_l2.yaml          strongly_convex_logistic
configs/non_convex/matrix_completion.yaml         non_convex_mc
configs/non_convex/phase_retrieval.yaml           non_convex_phase
configs/neural_networks/mlp_synthetic.yaml        nn_mlp
configs/inexact_hessian/logistic_delta.yaml       inexact_lr_delta
configs/scaling/lr_n_sweep.yaml                   scaling_lr
configs/scaling/lr_n200.yaml                      scaling_lr_n200
configs/scaling/lr_n800.yaml                      scaling_lr_n800
configs/oracle_cost_regime/expensive_grad.yaml    oracle_cost_regime
configs/ablations/rho_sweep.yaml                  ablation_rho
configs/ablations/rho_diagnostic.yaml             ablation_rho_diagnostic
configs/ablations/theta_sweep.yaml                ablation_theta
configs/ablations/gamma0_sweep.yaml               ablation_gamma0
configs/ablations/fp_max_sweep.yaml               ablation_fp_max
configs/ablations/restart_mode_sweep.yaml         ablation_restart
configs/ablations/momentum_offset_sweep.yaml      ablation_momentum
configs/ablations/warm_start_sweep.yaml           ablation_warm_start
configs/ablations/inexact_rank_compare.yaml       ablation_inexact_rank
configs/real_world/lr_a9a.yaml                    real_lr_a9a
configs/real_world/lr_w8a.yaml                    real_lr_w8a
configs/real_world/lr_phishing.yaml               real_lr_phishing
configs/real_world/svm_mushrooms.yaml             real_svm_mushrooms
configs/real_world/ridge_cadata.yaml              real_ridge_cadata
configs/real_world/lr_covtype.yaml                real_lr_covtype
EOF

# ---------------------------------------------------------------------------
# Step 1: data (unless explicitly skipped).
# ---------------------------------------------------------------------------

if [[ -z "${AGNS_SKIP_DATA}" && -z "${AGNS_SKIP_RUN}" ]]; then
    log "step 1: download_data.sh (set AGNS_SKIP_DATA=1 to skip)"
    run "bash '${SCRIPT_DIR}/download_data.sh'"
else
    log "step 1: skipped (AGNS_SKIP_DATA / AGNS_SKIP_RUN set)"
fi

# ---------------------------------------------------------------------------
# Step 2: campaigns.
# ---------------------------------------------------------------------------

while IFS= read -r row; do
    row="$(printf '%s' "${row}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "${row}" || "${row}" == \#* ]] && continue

    # Parse "<config> <name> [<nu_ref>]".  POSIX-portable.
    IFS=$' \t' read -r -a parts <<< "${row}"
    config="${REPO_ROOT}/${parts[0]}"
    campaign="${parts[1]}"
    nu_ref="${parts[2]:-}"

    if [[ ! -f "${config}" ]]; then
        log "WARN: missing config ${config}; skipping campaign ${campaign}"
        continue
    fi

    raw_dir="${NUMERICAL_RAW}/${campaign}"
    agg_path="${NUMERICAL_AGG}/${campaign}.json"
    fig_dir="${FIGURES_DIR}/${campaign}"
    tbl_path="${TABLES_DIR}/${campaign}.tex"

    log "== ${campaign}: ${config} =="

    if [[ -z "${AGNS_SKIP_RUN}" ]]; then
        log "  run_benchmark (seeds: ${AGNS_SEEDS})"
        run "${AGNS_PYTHON} -m agns.cli.run_benchmark \\
                --config '${config}' \\
                --output-dir '${raw_dir}' \\
                --seeds ${AGNS_SEEDS}"

        log "  aggregate"
        run "${AGNS_PYTHON} -m agns.cli.aggregate \\
                --campaign '${campaign}' \\
                --raw-dir '${raw_dir}' \\
                --output '${agg_path}'"
    else
        log "  run_benchmark + aggregate skipped (AGNS_SKIP_RUN); reusing ${agg_path}"
    fi

    nu_ref_arg=""
    if [[ -n "${nu_ref}" ]]; then
        nu_ref_arg="--nu-ref ${nu_ref}"
    fi
    # The rho sweep is the report's only main-body figure; render it
    # compact so the strict 3-page main body does not overflow.
    fig_size_arg=""
    if [[ "${campaign}" == "ablation_rho" ]]; then
        fig_size_arg="--figsize 6 2.8"
    fi
    log "  make_figures"
    run "${AGNS_PYTHON} -m agns.cli.make_figures \\
            --aggregated '${agg_path}' \\
            --output-dir '${fig_dir}' \\
            --no-title ${nu_ref_arg} ${fig_size_arg}"

    log "  make_tables"
    run "${AGNS_PYTHON} -m agns.cli.make_tables \\
            --aggregated '${agg_path}' \\
            --output '${tbl_path}'"
done <<< "${CAMPAIGNS}"

# ---------------------------------------------------------------------------
# Step 3: cross-dataset rollup table over the real_* campaigns.
# ---------------------------------------------------------------------------

log "step 3: real_world_summary.tex (cross-dataset rollup)"
run "${AGNS_PYTHON} -m agns.cli.make_summary \\
        --aggregated-dir '${NUMERICAL_AGG}' \\
        --output '${TABLES_DIR}/real_world_summary.tex'"

# ---------------------------------------------------------------------------
# Step 4: cross-campaign report figures (Picard diagnostic and the
# wall-clock-vs-n scaling plot) -- written under results/paper_figures/.
# ---------------------------------------------------------------------------

log "step 4: report figures (cross-campaign)"
run "${AGNS_PYTHON} -m agns.cli.make_figures \\
        --report-figures \\
        --results-dir '${AGNS_RESULTS}'"

log "done -- artefacts under ${AGNS_RESULTS}"
log "  numerical: ${NUMERICAL_RAW} and ${NUMERICAL_AGG}"
log "  figures:   ${FIGURES_DIR}"
log "  tables:    ${TABLES_DIR}"
