{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Search Content Decline Risk\n",
    "**Capstone, Machine Learning track. Lane: Refresh / Content Opportunity Scoring.**\n",
    "\n",
    "Question: *which pages are likely to lose search clicks next, so editors can review them first?*\n",
    "Run every cell top to bottom in Google Colab. No column names need editing: the schema is already mapped.\n",
    "Only the settings in Step 3 are adjustable."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1. Download the release files\n",
    "Files are saved to `data/` (ignored by git). Raw data is never committed or published."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!pip -q install duckdb scikit-learn matplotlib pandas pyarrow\n",
    "import os, urllib.request\n",
    "BASE = \"https://huggingface.co/buckets/stanley-metone/internship-warehouse-bucket/resolve/\"\n",
    "FILES = {\"clients\": \"dim_clients.parquet\", \"content\": \"dim_content.parquet\",\n",
    "         \"fact\": \"fact_content_daily_performance_sample.parquet\", \"query\": \"fact_content_query_90d.parquet\"}\n",
    "os.makedirs(\"data\", exist_ok=True)\n",
    "tok = os.environ.get(\"HF_TOKEN\")  # optional: only if the bucket asks for authentication\n",
    "for k, f in FILES.items():\n",
    "    p = f\"data/{f}\"\n",
    "    if not os.path.exists(p):\n",
    "        req = urllib.request.Request(BASE + f + \"?download=true\", headers={\"Authorization\": f\"Bearer {tok}\"} if tok else {})\n",
    "        with urllib.request.urlopen(req) as r, open(p, \"wb\") as o: o.write(r.read())\n",
    "    print(k, p, round(os.path.getsize(p) / 1e6, 1), \"MB\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2. Data quality: what can we trust?\n",
    "The daily table has one row per page per day, including days with no Search Console data\n",
    "(`gsc_data_available = False`). Those zeros are *missing*, not real zero clicks, so they are excluded."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import duckdb, numpy as np, pandas as pd\n",
    "con = duckdb.connect()\n",
    "P = lambda k: f\"read_parquet('data/{FILES[k]}')\"\n",
    "print(con.sql(f\"\"\"SELECT count(*) AS rows, count(DISTINCT client_hash_id) AS clients, count(DISTINCT content_hash_id) AS pages,\n",
    "    min(report_date) AS first_day, max(report_date) AS last_day,\n",
    "    round(avg(CASE WHEN gsc_data_available THEN 1 ELSE 0 END), 3) AS share_rows_with_gsc FROM {P('fact')}\"\"\").df().to_string(index=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3. Settings"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "MIN_CLICKS = 30        # a page needs this many clicks in the feature window to be scored\n",
    "MIN_AVAIL_DAYS = 7     # ...and at least this many days with Search Console data in the feature window\n",
    "MIN_LABEL_DAYS = 3     # ...and this many days with data in the label window\n",
    "DECLINE_DROP = 0.20    # label = daily clicks fell by at least 20% in the label window\n",
    "FEATURE_SHARE = 2 / 3  # first 2/3 of the days = feature window, last 1/3 = label window\n",
    "N_FOLDS = 5            # grouped cross-validation folds (split by client)\n",
    "SEED = 42"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4. Build the page-level panel (time-aware)\n",
    "Features come only from the feature window (the past). The label comes from the later window (the future).\n",
    "Extra features: page attributes from `dim_content`, and May query-level history (`*_prev30` columns only,\n",
    "which all end before the daily window starts, so they cannot leak the label)."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from datetime import timedelta\n",
    "con.sql(f\"\"\"CREATE OR REPLACE VIEW f AS SELECT client_hash_id cl, content_hash_id ct, CAST(report_date AS DATE) dt,\n",
    "  gsc_data_available av, ga4_data_available gav, gsc_impressions imp, gsc_clicks clk, gsc_avg_position pos,\n",
    "  ga4_sessions ses, ga4_engaged_sessions eng, sessions_organic org FROM {P('fact')}\"\"\")\n",
    "dmin, dmax = con.sql(\"SELECT min(dt), max(dt) FROM f WHERE av\").fetchone()\n",
    "span = (dmax - dmin).days + 1\n",
    "cut = dmin + timedelta(days=int(span * FEATURE_SHARE) - 1)\n",
    "F_days, L_days = (cut - dmin).days + 1, (dmax - cut).days\n",
    "print(\"GSC window:\", dmin, \"->\", dmax, \"| feature window ends:\", cut, \"| feature days:\", F_days, \"| label days:\", L_days)\n",
    "\n",
    "panel = con.sql(f\"\"\"\n",
    "WITH feat AS (\n",
    "  SELECT cl, ct, sum(clk) clicks, sum(imp) imps, count(DISTINCT dt) avail_days,\n",
    "    sum(pos*imp) / NULLIF(sum(CASE WHEN pos IS NOT NULL THEN imp END), 0) avg_pos,\n",
    "    sum(CASE WHEN dt > DATE '{cut}' - 7 THEN clk END) last7,\n",
    "    sum(CASE WHEN dt <= DATE '{cut}' - 7 AND dt > DATE '{cut}' - 14 THEN clk END) prev7,\n",
    "    sum(CASE WHEN gav THEN org END) org_sessions,\n",
    "    sum(CASE WHEN gav THEN eng END) / NULLIF(sum(CASE WHEN gav THEN ses END), 0) eng_rate\n",
    "  FROM f WHERE dt <= DATE '{cut}' AND av GROUP BY 1, 2),\n",
    "lab AS (SELECT cl, ct, sum(clk) lab_clicks, count(DISTINCT dt) lab_days FROM f WHERE dt > DATE '{cut}' AND av GROUP BY 1, 2)\n",
    "SELECT feat.*, lab.lab_clicks, lab.lab_days FROM feat JOIN lab USING (cl, ct)\n",
    "WHERE feat.clicks >= {MIN_CLICKS} AND feat.avail_days >= {MIN_AVAIL_DAYS} AND lab.lab_days >= {MIN_LABEL_DAYS}\"\"\").df()\n",
    "\n",
    "qf = con.sql(f\"\"\"SELECT client_hash_id cl, content_hash_id ct, sum(impressions_prev30) q_imps_prev, sum(clicks_prev30) q_clicks_prev,\n",
    "  sum(avg_position_prev30 * impressions_prev30) / NULLIF(sum(CASE WHEN avg_position_prev30 IS NOT NULL THEN impressions_prev30 END), 0) q_pos_prev,\n",
    "  count(*) FILTER (WHERE impressions_prev30 > 0) q_n_queries FROM {P('query')} GROUP BY 1, 2\"\"\").df()\n",
    "dc = con.sql(f\"\"\"SELECT client_hash_id cl, content_hash_id ct, word_count, search_volume, competition, cpc, backlinks,\n",
    "  keyword_token_count, category_count, main_intent, content_type, content_created_date FROM {P('content')}\n",
    "  WHERE NOT is_deleted AND is_published\"\"\").df()\n",
    "panel = panel.merge(qf, on=[\"cl\", \"ct\"], how=\"left\").merge(dc, on=[\"cl\", \"ct\"], how=\"inner\")\n",
    "panel[\"age_days\"] = (pd.Timestamp(cut) - pd.to_datetime(panel.content_created_date)).dt.days\n",
    "panel = panel[panel.age_days >= 0].reset_index(drop=True)   # drop pages created after the feature window ended\n",
    "\n",
    "panel[\"ctr\"] = panel.clicks / panel.imps.clip(lower=1)\n",
    "panel[\"momentum\"] = (panel.last7.fillna(0) + 1) / (panel.prev7.fillna(0) + 1)\n",
    "panel[\"feat_cpd\"] = panel.clicks / panel.avail_days\n",
    "panel[\"lab_cpd\"] = panel.lab_clicks / panel.lab_days\n",
    "panel[\"label\"] = (panel.lab_cpd <= (1 - DECLINE_DROP) * panel.feat_cpd).astype(int)\n",
    "print(panel.shape, \"| clients:\", panel.cl.nunique(), \"| decline rate:\", round(panel.label.mean(), 3))\n",
    "assert panel.label.nunique() == 2, \"label has one class: loosen MIN_CLICKS or DECLINE_DROP\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5. Baseline vs model on the same grouped folds\n",
    "Grouped 5-fold cross-validation: every client is held out exactly once, so no client appears in both\n",
    "train and test within a fold. Baseline = \"recent momentum is falling\" (lowest last-7 / prev-7 ratio ranks first)."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from sklearn.model_selection import GroupKFold\n",
    "from sklearn.ensemble import HistGradientBoostingClassifier\n",
    "from sklearn.metrics import roc_auc_score, average_precision_score\n",
    "NUM = [\"clicks\", \"imps\", \"avail_days\", \"avg_pos\", \"ctr\", \"momentum\", \"last7\", \"prev7\", \"org_sessions\", \"eng_rate\",\n",
    "       \"q_imps_prev\", \"q_clicks_prev\", \"q_pos_prev\", \"q_n_queries\", \"word_count\", \"search_volume\", \"competition\", \"cpc\",\n",
    "       \"backlinks\", \"keyword_token_count\", \"category_count\", \"age_days\"]\n",
    "CAT = [\"main_intent\", \"content_type\"]\n",
    "FEATS = NUM + CAT\n",
    "X = panel[NUM].astype(float).copy()\n",
    "for c in [\"clicks\", \"imps\", \"q_imps_prev\", \"q_clicks_prev\", \"search_volume\", \"backlinks\"]: X[c] = np.log1p(X[c].clip(lower=0))\n",
    "for c in CAT:\n",
    "    codes = panel[c].astype(\"category\").cat.codes.astype(float); X[c] = codes.where(codes >= 0, np.nan)\n",
    "cat_mask = [f in CAT for f in FEATS]\n",
    "y, g = panel.label.values, panel.cl.values\n",
    "p_base = -np.log(panel.momentum.values)       # falling momentum -> higher decline risk\n",
    "oof = np.zeros(len(y)); fold_id = np.zeros(len(y), int); fold_rows = []\n",
    "def prec_at(yt, s, frac=0.05):\n",
    "    k = max(1, int(len(s) * frac)); return yt[np.argsort(-s)[:k]].mean()\n",
    "for i, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, g)):\n",
    "    assert set(g[tr]).isdisjoint(set(g[te])), \"clients overlap across train/test\"\n",
    "    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, categorical_features=cat_mask, random_state=SEED).fit(X.iloc[tr], y[tr])\n",
    "    oof[te] = m.predict_proba(X.iloc[te])[:, 1]; fold_id[te] = i\n",
    "    fold_rows.append({\"fold\": i, \"n_test\": len(te), \"PR_model\": average_precision_score(y[te], oof[te]), \"PR_base\": average_precision_score(y[te], p_base[te])})\n",
    "    last = (m, tr, te)\n",
    "folds = pd.DataFrame(fold_rows); display(folds.round(3))\n",
    "res = pd.DataFrame({n: {\"ROC-AUC\": roc_auc_score(y, s), \"PR-AUC\": average_precision_score(y, s), \"Precision@top5%\": prec_at(y, s)}\n",
    "                    for n, s in [(\"Baseline\", p_base), (\"Model\", oof)]}).T\n",
    "res[\"Base rate\"] = y.mean(); display(res.round(3))\n",
    "pr_fold_sd = folds.PR_model.std()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6. Leakage checks"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "assert set(FEATS).isdisjoint({\"lab_clicks\", \"lab_days\", \"lab_cpd\", \"label\"}), \"label-window columns must not be features\"\n",
    "assert str(con.sql(f\"SELECT max(dt) FROM f WHERE dt <= DATE '{cut}' AND av\").fetchone()[0]) <= str(cut)\n",
    "assert (panel.age_days >= 0).all()\n",
    "m, tr, te = last\n",
    "rng = np.random.default_rng(SEED)\n",
    "shuf = HistGradientBoostingClassifier(max_depth=4, categorical_features=cat_mask, random_state=SEED).fit(X.iloc[tr], rng.permutation(y[tr]))\n",
    "shuf_auc = roc_auc_score(y[te], shuf.predict_proba(X.iloc[te])[:, 1])\n",
    "print(\"Shuffled-label AUC (should be near 0.5):\", round(shuf_auc, 3))\n",
    "print(\"Suspiciously strong single feature? max |corr with label|:\",\n",
    "      round(X.fillna(X.median()).corrwith(pd.Series(y)).abs().max(), 3), \"(values above ~0.8 deserve a second look)\")\n",
    "print(\"All leakage checks passed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 7. Charts (saved to `docs/figures/` for the paper)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import matplotlib; import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import precision_recall_curve\n",
    "from sklearn.inspection import permutation_importance\n",
    "os.makedirs(\"docs/figures\", exist_ok=True)\n",
    "plt.rcParams.update({\"axes.spines.top\": False, \"axes.spines.right\": False, \"font.size\": 10})\n",
    "# 1) precision-recall\n",
    "fig, ax = plt.subplots(figsize=(6, 4))\n",
    "for n, s in [(\"Baseline (momentum)\", p_base), (\"Model\", oof)]:\n",
    "    pr, rc, _ = precision_recall_curve(y, s); ax.plot(rc, pr, label=n)\n",
    "ax.axhline(y.mean(), ls=\"--\", c=\"grey\", label=\"Base rate\"); ax.set(xlabel=\"Recall\", ylabel=\"Precision\", title=\"Decline risk: model vs baseline (held-out clients)\")\n",
    "ax.legend(); fig.tight_layout(); fig.savefig(\"docs/figures/pr_curve.png\", dpi=160); plt.close(fig)\n",
    "# 2) permutation importance (last fold's held-out clients)\n",
    "pi = permutation_importance(m, X.iloc[te], y[te], n_repeats=3, random_state=SEED, scoring=\"average_precision\")\n",
    "s = pd.Series(pi.importances_mean, index=FEATS).sort_values().tail(12)\n",
    "fig, ax = plt.subplots(figsize=(6, 4.5)); s.plot.barh(ax=ax); ax.set_title(\"Top features (permutation, PR-AUC drop)\")\n",
    "fig.tight_layout(); fig.savefig(\"docs/figures/feature_importance.png\", dpi=160); plt.close(fig)\n",
    "# 3) lift by risk decile\n",
    "dec = pd.qcut(pd.Series(oof).rank(method=\"first\"), 10, labels=False)\n",
    "lift = pd.Series(y).groupby(dec).mean()\n",
    "fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(lift.index + 1, lift.values); ax.axhline(y.mean(), ls=\"--\", c=\"grey\")\n",
    "ax.set(xlabel=\"Predicted-risk decile (10 = highest risk)\", ylabel=\"Share that actually declined\", title=\"Pages in higher deciles decline more often\")\n",
    "fig.tight_layout(); fig.savefig(\"docs/figures/lift_by_decile.png\", dpi=160); plt.close(fig)\n",
    "# 4) decline rate by search position\n",
    "bins = pd.cut(panel.avg_pos, [0, 3, 10, 20, 1000], labels=[\"1-3\", \"4-10\", \"11-20\", \"21+\"])\n",
    "pos_rate = pd.Series(y).groupby(bins.values, observed=True).mean()\n",
    "fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(pos_rate.index.astype(str), pos_rate.values); ax.axhline(y.mean(), ls=\"--\", c=\"grey\")\n",
    "ax.set(xlabel=\"Average search position (feature window)\", ylabel=\"Share that declined\", title=\"Decline rate by search position\")\n",
    "fig.tight_layout(); fig.savefig(\"docs/figures/decline_by_position.png\", dpi=160); plt.close(fig)\n",
    "print(\"saved 4 charts\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 8. Ranked recommendations with reason codes\n",
    "`playbook_summary.csv` is aggregate and safe to publish. `ranked_pages_PRIVATE.csv` holds hashed page IDs for\n",
    "your own review queue: **do not commit or publish it** (the folder is git-ignored)."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "test = panel.copy(); test[\"risk\"] = oof\n",
    "q = test.risk.quantile(0.95)\n",
    "def codes(r):\n",
    "    out = []\n",
    "    if r.momentum < 0.7: out.append(\"MOMENTUM_DOWN\")\n",
    "    if pd.notna(r.avg_pos) and r.avg_pos > 10: out.append(\"LOW_RANK_POSITION\")\n",
    "    if r.ctr < test.ctr.median() * 0.7: out.append(\"CTR_BELOW_MEDIAN\")\n",
    "    if r.avail_days < F_days * 0.6: out.append(\"INTERMITTENT_VISIBILITY\")\n",
    "    return out or [\"MODEL_ONLY\"]\n",
    "top = test[test.risk >= q].copy(); top[\"reasons\"] = top.apply(codes, axis=1)\n",
    "playbook = (top.explode(\"reasons\").groupby(\"reasons\").agg(pages=(\"cl\", \"size\"), avg_risk=(\"risk\", \"mean\"), avg_daily_clicks=(\"feat_cpd\", \"mean\"))\n",
    "            .sort_values(\"pages\", ascending=False).round(3))\n",
    "display(playbook)\n",
    "os.makedirs(\"outputs\", exist_ok=True)\n",
    "playbook.to_csv(\"outputs/playbook_summary.csv\"); res.round(4).to_csv(\"outputs/results.csv\"); folds.round(4).to_csv(\"outputs/folds.csv\")\n",
    "top.sort_values(\"risk\", ascending=False).head(200)[[\"cl\", \"ct\", \"risk\", \"feat_cpd\", \"momentum\", \"avg_pos\", \"ctr\", \"reasons\"]].to_csv(\"outputs/ranked_pages_PRIVATE.csv\", index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 9. Limitations (already written into the paper)\n",
    "- Observed associations and directional signals, not causal claims; nothing here proves how Google ranks pages.\n",
    "- The daily file covers one month, so \"decline\" is a short-horizon label; a 10-day label window is noisy.\n",
    "- Page attributes in `dim_content` are a snapshot, not point-in-time values; May query history is partial (rare queries are anonymized).\n",
    "- AI-referral sessions are too sparse to model here and were left out on purpose."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 10. Fill the paper with your real numbers\n",
    "Run last, after `docs/index.html` is uploaded to your repo. It downloads the finished paper and the four charts."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "RAW = \"https://raw.githubusercontent.com/stanley-metone/search-content-decline-risk/main/docs/index.html\"\n",
    "html = urllib.request.urlopen(RAW).read().decode()   # needs docs/index.html already uploaded to your repo\n",
    "f3 = lambda v: f\"{v:.3f}\"\n",
    "pb = playbook[\"pages\"].to_dict()\n",
    "vals = {\n",
    " \"N_PAGES\": f\"{len(panel):,}\", \"N_CLIENTS\": f\"{panel.cl.nunique():,}\", \"DATE_RANGE\": f\"{dmin} to {dmax}\", \"SHUF_AUC\": f3(shuf_auc),\n",
    " \"ROC_BASE\": f3(res.loc[\"Baseline\", \"ROC-AUC\"]), \"PR_BASE\": f3(res.loc[\"Baseline\", \"PR-AUC\"]), \"PR_BASE_T\": f3(res.loc[\"Baseline\", \"PR-AUC\"]),\n",
    " \"P5_BASE\": f3(res.loc[\"Baseline\", \"Precision@top5%\"]), \"ROC_MODEL\": f3(res.loc[\"Model\", \"ROC-AUC\"]),\n",
    " \"PR_MODEL\": f3(res.loc[\"Model\", \"PR-AUC\"]), \"PR_MODEL_T\": f3(res.loc[\"Model\", \"PR-AUC\"]), \"P5_MODEL\": f3(res.loc[\"Model\", \"Precision@top5%\"]),\n",
    " \"BASE_RATE\": f3(y.mean()), \"PR_FOLD_SD\": f3(pr_fold_sd), \"LIFT5\": f\"{res.loc['Model', 'Precision@top5%'] / y.mean():.1f}x\",\n",
    " \"C_MOM\": pb.get(\"MOMENTUM_DOWN\", 0), \"C_POS\": pb.get(\"LOW_RANK_POSITION\", 0),\n",
    " \"C_CTR\": pb.get(\"CTR_BELOW_MEDIAN\", 0), \"C_INT\": pb.get(\"INTERMITTENT_VISIBILITY\", 0)}\n",
    "for k, v in vals.items():\n",
    "    html = html.replace('<span class=\"fill\">{{%s}}</span>' % k, str(v))\n",
    "assert \"{{\" not in html, \"some placeholders were not filled\"\n",
    "open(\"docs/index.html\", \"w\").write(html)\n",
    "try:\n",
    "    from google.colab import files\n",
    "    for fn in [\"docs/index.html\", \"docs/figures/pr_curve.png\", \"docs/figures/feature_importance.png\",\n",
    "               \"docs/figures/lift_by_decile.png\", \"docs/figures/decline_by_position.png\"]: files.download(fn)\n",
    "except ImportError:\n",
    "    print(\"Not in Colab: files are saved in docs/\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}