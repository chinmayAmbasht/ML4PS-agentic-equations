#!/usr/bin/env bash
# Regenerate the paper outputs. Run from the repository root.
#   ./reproduce.sh quick     ~1 min   Table 1, Table 2 (all 64 forms), the ranking and Fig. 3 from the stored results
#   ./reproduce.sh figures   minutes  + Fig. 1, the covariance and ablation appendix figures, the held-out fits appendix
#   ./reproduce.sh full      CPU-hours re-score all 64 forms and recompute every loss from the processed data, then 'figures'
# The PySR searches (PySR/run_all.sh, 10 x 45 min) and the ablation arms (python -m Ablation.run_arm A1|A2|A3) are long
# and are not run by 'full'; their stored outputs are used instead.
set -euo pipefail
cd "$(dirname "$0")"
export ML4PS_CACHE="${ML4PS_CACHE:-$PWD/cache}"
FORMS="T47 T36 T43 T39 T51"     # the five forms of Table 1 / Fig. 3 (ranks 1, 2, 3, 10, 49)
mode="${1:-quick}"

quick() {
  python -m PaperUpdate.rank_all
  python -m PaperUpdate.table1_main > /dev/null
  python -m PaperUpdate.make_tex
  python -m PaperUpdate.figs.fig3_benchmark $FORMS
}
figures() {
  python -m PaperUpdate.figs.fig1_oldstyle
  python -m PaperUpdate.figs.fig_cfv_appendix
  python -m Ablation.analyze
  python -m Ablation.summary_fig
  python -m PaperUpdate.figs.fig_fits_appendix $FORMS
}
full() {
  python -m PaperUpdate.sigma_k_hartlap
  python -m PaperUpdate.rescore_all all          # Stage 1/2, log Z_cross ladder, chi2 and per-box predictions for all 64 forms
  python -m PaperUpdate.Lcorr_all                # correlated loss L for every form
  python -m PaperUpdate.Benchmark.bench $FORMS   # within / pooled / transfer losses, random forest, van Daalen
  python -m PaperUpdate.vd_pipeline 11           # van Daalen+20 through the Stage-2 + per-box MAP pipeline
  python -m PaperUpdate.Benchmark.attach_pipeline "vD20 refit"=VD
  python -m PySR.pipeline U C                    # PySR pooled equations through the harness
  for g in loop pysr vd; do python -m PaperUpdate.map_sensitivity "$g"; done
  python -m PaperUpdate.apply_map_setting        # Fig. 3 adapted bars: unwidened prior, C_fv/h likelihood
}

case "$mode" in
  quick) quick ;;
  figures) quick; figures ;;
  full) full; quick; figures ;;
  *) echo "usage: $0 [quick|figures|full]"; exit 1 ;;
esac
echo "done ($mode). Figures: PaperUpdate/figs/out/. Tables: outputs/tables/, Ablation/ablation_table.tex."
