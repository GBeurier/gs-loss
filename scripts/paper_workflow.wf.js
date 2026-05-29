export const meta = {
  name: 'ccgp-paper',
  description: 'Draft Results/Discussion/Abstract from the analysis tables, adversarially peer-review as G3 reviewers, then revise',
  phases: [
    { title: 'Draft', detail: 'parallel section drafting grounded in results/analysis/*' },
    { title: 'Review', detail: 'three critical G3 reviewers' },
    { title: 'Revise', detail: 'address concerns, stay grounded in the numbers' },
  ],
}

const ROOT = '/home/delete/genomics'
const SECTION_SCHEMA = {
  type: 'object',
  properties: {
    latex: { type: 'string', description: 'LaTeX body prose only (no preamble)' },
    key_numbers: { type: 'array', items: { type: 'string' } },
  },
  required: ['latex'],
}

const context = `You are writing part of a G3 (Genes|Genomes|Genetics) methods paper. READ these files first and ground EVERY quantitative claim in their actual contents (do not invent numbers):
- ${ROOT}/METHODS_NOTES.md  (the authoritative methods spec + verified facts)
- ${ROOT}/Overview.md  (the intended narrative)
- ${ROOT}/results/exp_a.json  (numerical verification)
- ${ROOT}/results/analysis/deltas_summary.csv  (Delta vs MSE per loss/model/calibration, bootstrap CIs, Holm/FDR p)
- ${ROOT}/results/analysis/calibration_summary.csv  (raw/affine/isotonic effect on Pearson-trained NN)
- ${ROOT}/results/analysis/ranks.csv  (mean method ranks)
- ${ROOT}/results/analysis/lmm_*.csv  (mixed-model coefficients)
- ${ROOT}/paper/main.tex  (current skeleton; match its \\label and \\citep keys)
Use \\citep{key} with keys from ${ROOT}/paper/refs.bib. Reference figures by their existing labels (fig:geometry, fig:datasets, fig:loss, fig:calibration, fig:meta). Be precise, honest, and explicit about where the Pearson loss helps and where it does not. Output LaTeX body prose only.`

phase('Draft')
const sections = [
  { key: 'results', prompt: `Write the Results section body (subsections \\subsection{...} matching the skeleton's 4.1-4.7 labels: res-numerical, res-loss, res-calibration, res-selection, res-splits, res-batch, res-meta). Report the loss comparison with concrete Delta values, bootstrap CIs and corrected p-values; the calibration effect (Pearson unchanged, RMSE/slope/intercept improved); selection-oriented outcomes; robustness across random/family/cross-env splits; the batch-size finding; and the mixed-model meta-analysis.` },
  { key: 'discussion', prompt: `Write the Discussion: (1) the Pearson loss is not a universal winner; (2) calibration is mandatory when phenotypic scale matters; (3) selection requires top-k/ranking metrics; (4) concrete recommendations for breeders; (5) limitations (including the HPO-on-full-data caveat in METHODS_NOTES) and future work.` },
  { key: 'abstract', prompt: `Write a ~200-word abstract for the article with the actual headline numbers filled in.` },
]
const drafts = (await parallel(sections.map(s => () =>
  agent(`${context}\n\nTASK: ${s.prompt}`, { label: `draft:${s.key}`, phase: 'Draft', schema: SECTION_SCHEMA })
    .then(r => ({ key: s.key, ...r }))))).filter(Boolean)

phase('Review')
const combined = drafts.map(d => `%%% ${d.key}\n${d.latex}`).join('\n\n')
const lenses = [
  'methodological and statistical rigor: leakage, multiple-testing correction, validity of the mixed model, whether the equivalence and calibration claims are correctly stated',
  'novelty and positioning against prior work (Waldmann 2019, Blondel 2015, Ornella 2014, PNNGS 2024, Montesinos-Lopez reviews); is the contribution clearly delineated and not overclaimed',
  'clarity and internal consistency: does every stated number match results/analysis/, are any claims overstated relative to the data',
]
const REVIEW_SCHEMA = { type: 'object', properties: {
  concerns: { type: 'array', items: { type: 'string' } },
  overstated_claims: { type: 'array', items: { type: 'string' } },
  verdict: { type: 'string', enum: ['accept', 'minor revision', 'major revision', 'reject'] },
}, required: ['concerns', 'verdict'] }
const reviews = (await parallel(lenses.map((l, i) => () =>
  agent(`You are a critical but fair G3 reviewer. First READ ${ROOT}/METHODS_NOTES.md and the CSVs in ${ROOT}/results/analysis/. Then review the draft below through this lens: ${l}. Give concrete, actionable concerns and flag any claim not supported by the actual numbers.\n\nDRAFT:\n${combined}`,
    { label: `review:${i + 1}`, phase: 'Review', schema: REVIEW_SCHEMA })))).filter(Boolean)

phase('Revise')
const critiques = reviews.flatMap(r => [...(r.concerns || []), ...(r.overstated_claims || [])]).join('\n- ')
const revised = (await parallel(sections.map(s => () => {
  const d = drafts.find(x => x.key === s.key)
  return agent(`Revise this ${s.key} LaTeX to address the reviewer concerns, staying strictly grounded in the numbers in ${ROOT}/results/analysis/. Do not overclaim. Return improved LaTeX body only.\n\nREVIEWER CONCERNS:\n- ${critiques}\n\nCURRENT ${s.key}:\n${d ? d.latex : ''}`,
    { label: `revise:${s.key}`, phase: 'Revise', schema: SECTION_SCHEMA }).then(r => ({ key: s.key, ...r }))
}))).filter(Boolean)

return {
  sections: revised.map(d => ({ key: d.key, latex: d.latex, key_numbers: d.key_numbers || [] })),
  reviews: reviews.map((r, i) => ({ lens: i + 1, verdict: r.verdict, concerns: r.concerns })),
}
