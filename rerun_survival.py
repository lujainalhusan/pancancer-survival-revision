"""Reruns only the censoring aware survival block after the stratified
prediction alignment fix; inputs are rebuilt exactly as in main()."""
import sys
import pancancer_revision as pr
out = sys.argv[3]
Xdf, clin, clin_full = pr.build_cohort(sys.argv[1], sys.argv[2], out + "/tmp_flow")
C_all = pr.clin_frame(clin)
pr.survival_analyses(Xdf.values, C_all.reset_index(drop=True), clin[pr.C_OST].values.astype(float),
                     clin[pr.C_OS].values.astype(int), C_all["type"].values, out, repeats=2)
pr.log("survival rerun done")
