export const meta = {
  name: 'g3-readiness-review',
  description: 'Adversarial G3 editorial review of the finalized manuscript: rigor, novelty/positioning, statistics, clarity/format/reproducibility -> consolidated readiness verdict + prioritized fix checklist',
  phases: [
    { title: 'Review', detail: 'four independent G3 reviewers, distinct lenses' },
    { title: 'Synthesize', detail: 'handling editor consolidates a verdict + checklist' },
  ],
}

const ROOT = '/home/delete/gs-loss'
const SRC = `READ the manuscript and evidence before judging:
- ${ROOT}/paper/main.tex (master: title, intro, Theory with Props 1-3, Methods, Conclusion, Table 1, figures)
- ${ROOT}/paper/abstract_body.tex, ${ROOT}/paper/sec_results.tex, ${ROOT}/paper/sec_sorghum.tex, ${ROOT}/paper/discussion_body.tex
- ${ROOT}/METHODS_NOTES.md (authoritative methods + verified facts)
- ${ROOT}/results/analysis/*.csv (deltas_summary, calibration_summary, ranks, splits_summary, batch_summary, lmm_*_affine) and ${ROOT}/results/exp_a.json
Cross-check that every quantitative claim in the text matches these files. The journal is G3: Genes|Genomes|Genetics (Genomic Selection article type); G3 judges technical soundness and reproducibility, not perceived impact.`

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['accept', 'minor revision', 'major revision', 'reject'] },
    strengths: { type: 'array', items: { type: 'string' } },
    must_fix: { type: 'array', items: { type: 'string', description: 'blocking issue with location + concrete fix' } },
    nice_to_have: { type: 'array', items: { type: 'string' } },
    unsupported_or_overstated_claims: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdict', 'strengths', 'must_fix'],
}

phase('Review')
const lenses = [
  { key: 'rigor', focus: 'Scientific rigor & correctness: do the conclusions follow from the data? leakage (HPO/calibration/standardization), the standardized-MSE and affine-calibration claims (is affine stated as EXACTLY rank/r-preserving when slope>0, with the <1% sign-flip caveat?), the architecture-dependence claim, the only-CCC-improves-RMSE claim, GBLUP validation. Flag any claim not supported by results/analysis/.' },
  { key: 'novelty', focus: 'Novelty & positioning for a Genomic Selection audience: is the contribution (empirical characterization + calibration framing) clearly delineated from the elementary identity and from prior work (Waldmann 2019, Blondel 2015, Ornella 2014, PNNGS 2024, Montesinos-Lopez reviews, Pandit&Schuller CCC-MSE)? Is anything overclaimed? Is it a good fit + sufficient advance for G3?' },
  { key: 'stats', focus: 'Statistical methodology: cell-level paired deltas vs the mixed model, bootstrap BCa CIs, Holm/FDR, the pseudoreplication handling, p-value-vs-CI consistency (esp. RMSE), the underpowered splits (n=4-12) framing, and whether multiple-testing and uncertainty are reported correctly per G3 norms.' },
  { key: 'clarity', focus: 'Clarity, structure, reproducibility, and G3 format compliance: required sections present (Abstract, Intro, Materials & Methods incl. Statistical Analysis + Data Availability, Results, Discussion); abstract single paragraph <250 words, no citations; data/code availability with the public repo + proprietary-sorghum note; figures/tables referenced and self-contained; reproducibility (public data, pinned env, fixed folds). Note any G3 style issues.' },
]
const reviews = (await parallel(lenses.map(l => () =>
  agent(`${SRC}\n\nYou are a critical but fair G3 reviewer. Review through this lens: ${l.focus}\nGive a verdict and concrete, actionable items (cite file/section).`,
    { label: `review:${l.key}`, phase: 'Review', schema: REVIEW_SCHEMA })
    .then(r => ({ lens: l.key, ...r }))))).filter(Boolean)

phase('Synthesize')
const packed = reviews.map(r => `### Reviewer (${r.lens}) — ${r.verdict}\nMUST-FIX:\n- ${(r.must_fix||[]).join('\n- ')}\nOVERSTATED:\n- ${(r.unsupported_or_overstated_claims||[]).join('\n- ')}\nNICE:\n- ${(r.nice_to_have||[]).join('\n- ')}`).join('\n\n')
const editor = await agent(
  `${SRC}\n\nYou are the G3 handling editor. Below are four reviewer reports. Read the manuscript yourself, then consolidate a single G3-readiness assessment: (1) an overall verdict (ready-to-submit / minor fixes / major fixes); (2) a deduplicated, PRIORITIZED checklist split into BLOCKING (must fix before submission), RECOMMENDED, and OPTIONAL; (3) a 4-6 sentence editor summary of the paper's standing for G3. Be decisive and concrete.\n\nREVIEWER REPORTS:\n${packed}`,
  { label: 'editor-synthesis', phase: 'Synthesize', schema: {
    type: 'object',
    properties: {
      overall_verdict: { type: 'string' },
      blocking: { type: 'array', items: { type: 'string' } },
      recommended: { type: 'array', items: { type: 'string' } },
      optional: { type: 'array', items: { type: 'string' } },
      editor_summary: { type: 'string' },
    },
    required: ['overall_verdict', 'blocking', 'editor_summary'],
  } })

return { editor, reviewer_verdicts: reviews.map(r => ({ lens: r.lens, verdict: r.verdict })) }
