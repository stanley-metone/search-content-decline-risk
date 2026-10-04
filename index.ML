<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Which Pages Decline Next? Scoring Content for Refresh Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,600;1,400&family=Public+Sans:wght@500;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f3f5f7;--ink:#17212b;--mute:#58656f;--rule:#cdd5db;--accent:#1f5f8b;--fill:#fff3bf}
@media (prefers-color-scheme:dark){:root{--bg:#12181e;--ink:#e6ebef;--mute:#9aa7b1;--rule:#2b3640;--accent:#6fb0dc;--fill:#4a3f10}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:19px/1.65 Newsreader,Georgia,serif;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
main{max-width:46rem;margin:0 auto;padding:3rem 1.25rem 5rem}
h1,h2,h3{font-family:"Public Sans",system-ui,sans-serif;line-height:1.15;letter-spacing:-.01em}
h1{font-size:2.3rem;margin:0 0 .5rem}
h2{font-size:1.3rem;margin:3rem 0 .6rem;padding-top:1rem;border-top:1px solid var(--rule)}
h3{font-size:1.02rem;margin:1.6rem 0 .3rem}
.sub{color:var(--mute);font-style:italic;margin:0 0 2rem}
.abstract{border-left:3px solid var(--accent);padding:.2rem 0 .2rem 1rem}
a{color:var(--accent)}
img{max-width:100%;height:auto;border:1px solid var(--rule);background:#fff}
figure{margin:1.4rem 0}
table{border-collapse:collapse;width:100%;font-size:.95rem}
th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--rule)}
.scroll{overflow-x:auto}
.fill{background:var(--fill);padding:0 .25em;border-radius:3px}
figcaption{color:var(--mute);font-size:.9rem}
code{font-size:.88em}
</style></head><body><main>
<h1>Which Pages Decline Next? Scoring Content for Refresh Review</h1>
<p class="sub">Capstone, Machine Learning track &middot; Stanley Metone</p>

<h2>Abstract</h2>
<p class="abstract">Content teams cannot review every page, so they need to know where to look first. We ask which pages are likely to lose search clicks in the following weeks. Using daily Search Console performance for <span class="fill">{{N_PAGES}}</span> pages across <span class="fill">{{N_CLIENTS}}</span> pseudonymized clients, we build features from an earlier window and label declines in a later window. A gradient-boosted model reaches PR-AUC <span class="fill">{{PR_MODEL}}</span> against <span class="fill">{{PR_BASE}}</span> for a momentum baseline on held-out clients, and its top 5% of ranked pages decline <span class="fill">{{LIFT5}}</span> more often than average. The output is a reason-coded review list, offered as decision support rather than proof of causation.</p>

<h2>Introduction</h2>
<p>The decision this work supports: with limited editor time, which pages should be reviewed for a refresh this cycle? A short ranked list with reasons is more useful than a full audit, provided it is better than the obvious shortcut of looking at whatever just dropped. We test exactly that: does a model using page, search and engagement signals beat a simple &ldquo;recent clicks are falling&rdquo; rule?</p>

<h2>Data</h2>
<p>Release v20260703 of the FlyRank pseudonymized warehouse. The main table is the daily page performance sample (<span class="fill">{{DATE_RANGE}}</span>), with page attributes from <code>dim_content</code> and May query-level history from <code>fact_content_query_90d</code>. All identifiers are hashed.</p>
<p><b>Exclusions.</b> Rows where Search Console data was unavailable were dropped, because those zeros are missing data rather than real zero clicks. Deleted and unpublished pages were dropped. Pages needed at least 30 clicks and 7 days of data in the feature window, and at least 3 days of data in the label window; below that, rates are too noisy to score. Pages created after the feature window ended were removed. AI-referral sessions were left out because they are too sparse to model reliably. No client names, domains, URLs or queries appear in this paper.</p>

<h2>Methodology</h2>
<h3>Windows and label</h3>
<p>The first two thirds of days form the feature window and the final third the label window, so features only describe the past. A page is labelled a <i>decline</i> if its average daily clicks fell by at least 20% between windows, measured over days with available data.</p>
<h3>Features</h3>
<p>From the feature window: clicks, impressions, click-through rate, impression-weighted position, days with data, last-7 versus prior-7-day click momentum, organic sessions and engagement rate. From May query history (columns ending <code>prev30</code> only, all before the daily window): impressions, clicks, position and number of visible queries. From page attributes: word count, search volume, competition, CPC, backlinks, keyword length, categories, intent, content type and age.</p>
<h3>Baseline and validation</h3>
<p>The baseline ranks pages by falling momentum alone. Both approaches are scored on the same grouped 5-fold cross-validation by client, so no client appears in both training and test within a fold.</p>
<h3>Leakage checks</h3>
<p>Label-window columns are excluded from the features, feature dates end at the window cut-off, train and test clients never overlap, and May query columns end before June begins. A model trained on shuffled labels scored AUC <span class="fill">{{SHUF_AUC}}</span>, close to the 0.5 expected from chance.</p>

<h2>Results</h2>
<div class="scroll"><table>
<tr><th>Approach</th><th>ROC-AUC</th><th>PR-AUC</th><th>Precision at top 5%</th></tr>
<tr><td>Baseline (momentum)</td><td><span class="fill">{{ROC_BASE}}</span></td><td><span class="fill">{{PR_BASE_T}}</span></td><td><span class="fill">{{P5_BASE}}</span></td></tr>
<tr><td>Gradient boosting</td><td><span class="fill">{{ROC_MODEL}}</span></td><td><span class="fill">{{PR_MODEL_T}}</span></td><td><span class="fill">{{P5_MODEL}}</span></td></tr>
</table></div>
<p>Share of pages that declined (the base rate): <span class="fill">{{BASE_RATE}}</span>. Model PR-AUC varied by <span class="fill">{{PR_FOLD_SD}}</span> (standard deviation) across folds.</p>
<figure><img src="figures/pr_curve.png" alt="Precision-recall curves for the model and baseline on held-out clients"><figcaption>Precision and recall for the model and the momentum baseline on held-out clients.</figcaption></figure>
<figure><img src="figures/lift_by_decile.png" alt="Share of pages that declined, by predicted-risk decile"><figcaption>Pages the model ranks as riskier decline more often, which is what a review queue needs.</figcaption></figure>
<figure><img src="figures/decline_by_position.png" alt="Decline rate by average search position"><figcaption>Decline rate by average search position in the feature window.</figcaption></figure>
<figure><img src="figures/feature_importance.png" alt="Permutation importance of the top features"><figcaption>Features the model relies on most, measured by the drop in PR-AUC when each is shuffled. Importance shows association, not cause.</figcaption></figure>

<h2>Limitations and honest framing</h2>
<p>These results are observed associations and directional signals for decision support. They do not prove how Google ranks pages, and they do not show that refreshing a page causes recovery. The daily sample covers one month, so the label is a short-horizon decline measured over a narrow window and is noisy. Page attributes are a snapshot rather than point-in-time values, and query history is partial because rare queries are anonymized. A drop can also reflect seasonality or measurement gaps. Use the ranked list to prioritise human review, not as an automatic instruction to rewrite.</p>

<h2>Ranked recommendations (action playbook)</h2>
<p>Pages in the top 5% of predicted decline risk, grouped by reason code. A page can carry more than one code.</p>
<div class="scroll"><table>
<tr><th>Reason code</th><th>Pages</th><th>What to do</th></tr>
<tr><td><code>MOMENTUM_DOWN</code></td><td><span class="fill">{{C_MOM}}</span></td><td>Review first for freshness, outdated facts and intent match.</td></tr>
<tr><td><code>LOW_RANK_POSITION</code></td><td><span class="fill">{{C_POS}}</span></td><td>Improve depth, internal links and answer quality; consider a rewrite.</td></tr>
<tr><td><code>CTR_BELOW_MEDIAN</code></td><td><span class="fill">{{C_CTR}}</span></td><td>Test title and meta description changes.</td></tr>
<tr><td><code>INTERMITTENT_VISIBILITY</code></td><td><span class="fill">{{C_INT}}</span></td><td>Monitor; check indexing, seasonality and data gaps before editing.</td></tr>
</table></div>
<p>Suggested workflow: review the high-risk list weekly, start with pages that have both a reason code and high daily clicks, and re-score after changes ship to see whether the trend moves.</p>

<h2>Reproducibility</h2>
<p>All code is in the <a href="https://github.com/stanley-metone/search-content-decline-risk">project repository</a>. Run <code>work/capstone.ipynb</code> top to bottom in Google Colab; it downloads the data, trains both approaches, runs the leakage checks and writes the charts and tables used here.</p>

<h2>Acknowledgments and data credit</h2>
<p>Built on the FlyRank ML Internship dataset, <a href="https://flyrank.ai" target="_blank" rel="noopener">https://flyrank.ai</a>. The data is pseudonymized and used for anonymized research and education only.</p>
</main></body></html>
