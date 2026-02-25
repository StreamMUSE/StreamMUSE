#!/usr/bin/env bash
# run_nll.sh — Batch NLL computation for all parameter combinations.
#
# Run this from the StreamMUSE-1 root directory AFTER test-run.sh has
# finished generating the MIDI outputs.
#
# Mirrors the same interval/gen_frame combinations as test-run.sh.
# Output JSONs are written directly into the eval repo's nll_runs/ folder
# so that add_nll_to_summary.py can discover them automatically.
#
# Usage:
#   bash nll_compute/run_nll.sh
#
# Adjust the variables below before running:

# ── CONFIG ────────────────────────────────────────────────────────────────────
EXP_ROOT="experiments-AE5"                         # must match --out-root prefix in test-run.sh
CKPT_PATH="/path/to/model.ckpt"                    # model checkpoint
PROMPT_GEN="prompt_128_gen_576"                    # must match injection/generation lengths used
EVAL_NLL_DIR="/path/to/eval/results-${EXP_ROOT}/nll_runs"  # destination in eval repo
# ⚠️ UNIT NOTE: core.py measures MIDI file length in MIDI ticks, not model frames.
# --generation-length 576 (model frames) produces ~288 MIDI ticks (1 tick = 2 frames).
# So --window must be < ~288. Use 256 to guarantee at least one sliding window pass.
WINDOW=256
OFFSET=64
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "${EVAL_NLL_DIR}"

run_nll() {
    local INTERVAL=$1
    local GEN_FRAME=$2
    local MIDI_DIR="${EXP_ROOT}/realtime/baseline/interval_${INTERVAL}_gen_frame_${GEN_FRAME}/${PROMPT_GEN}/generated"
    local OUT_JSON="${EVAL_NLL_DIR}/experiments_interval${INTERVAL}_gen${GEN_FRAME}.json"

    echo "▶ interval=${INTERVAL} gen_frame=${GEN_FRAME}"
    echo "  midi_dir : ${MIDI_DIR}"
    echo "  out_json : ${OUT_JSON}"

    if [ ! -d "${MIDI_DIR}" ]; then
        echo "  ⚠️  MIDI dir not found, skipping."
        return
    fi

    uv run python -m nll_compute.runners.run_cal_nll \
        --midi_dir   "${MIDI_DIR}" \
        --ckpt_path  "${CKPT_PATH}" \
        --save_json_path "${OUT_JSON}" \
        --window ${WINDOW} \
        --offset ${OFFSET}

    echo "  ✅ Done → ${OUT_JSON}"
    echo
}

# ── COMBINATIONS (same grid as test-run.sh) ───────────────────────────────────
run_nll 1 3
run_nll 2 5
run_nll 2 9
run_nll 4 5
run_nll 4 9
run_nll 4 15
run_nll 7 9
run_nll 7 15
# ──────────────────────────────────────────────────────────────────────────────

echo "All NLL computations complete."
echo "Results saved to: ${EVAL_NLL_DIR}"
echo "Next step: run  add_nll_to_summary.py  in the eval repo."
