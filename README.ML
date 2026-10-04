Search Content Decline Risk

Machine learning capstone (Refresh / Content Opportunity Scoring lane). Ranks web pages by their risk of losing search clicks so content teams know what to review first. Time-aware features and labels, grouped-by-client cross-validation against a momentum baseline, leakage checks, and reason-coded recommendations.

Paper: see submission/paper_url.txt

Repo layout
work/ holds every assignment notebook plus capstone.ipynb
docs/index.html is the research paper (served by GitHub Pages from /docs); charts are in docs/figures/
submission/paper_url.txt holds one line: the deployed paper URL
Reproduce
Open work/capstone.ipynb in Google Colab.
Run all cells. Raw data downloads to data/ and is never committed.
Outputs (charts, aggregate tables) are written to docs/figures/ and outputs/.
Public-safe note

No client names, domains, URLs, queries, credentials or raw exports are included. outputs/ (which holds the private review queue) is git-ignored.

Built on the FlyRank ML Internship dataset (https://flyrank.ai). Anonymized research and education use only.
