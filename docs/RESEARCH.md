# Adaptive Post-Timeout Verification under Stale Observations

**فارسی:** کاهش هزینه در این شبیه‌سازی مشروط به تابع زیان است؛ اعلام زودهنگام موفقیت می‌تواند رتبه‌بندی سیاست‌ها را عوض کند.

## Findings and limits
Adaptive baseline cost is 5.101, versus 12.280 for the development-tuned 8-second wait: 58.5% lower. Permanently false success occurs in 1.95% and premature success before a later commit in 29.35% of episodes. The latter is not charged by the baseline loss. Repricing frozen trajectories at 100 units per premature declaration gives adaptive cost 34.451, reversing its ranking against the tuned wait. This is not a retuned comparison under the new loss.

The new retry-loss and actual-sigma sensitivities use fresh seed streams and are exploratory. Only one retry is modeled. A missing retry can be avoided by escalation without completing the work autonomously. The outcome oracle is optimized for the baseline loss only. No finite verification-only completion policy gains an exactly-once guarantee without an appropriate contract/bound. All outcomes are parametric simulation, not production measurements.

## Reproduce
Use Python 3.12 and pinned requirements:

```bash
pip install -r requirements.txt
bash run_all.sh
docker build -t p3_post_timeout_verification .
docker run --name p3_reproduction p3_post_timeout_verification
docker cp p3_reproduction:/work/results ./container_results
docker cp p3_reproduction:/work/paper/supplementary_p3.zip ./
```

```powershell
./run_all.ps1
./run_all.ps1 -SkipPdf
```

The complete pipeline runs tests, all six simulation experiments, paired/Holm analysis, risk and repricing summaries, a labelled model-harness mock, manuscript construction, numeric/citation/template audits and supplementary packaging. Windows PDF export requires Word; Docker creates the DOCX and scientific outputs. No external benchmark was run.

## Seeds and statistics
Main 4000 episodes start at 0. Fixed-wait development uses 600 episodes starting at 900000 for each candidate. Retry-loss and actual-sigma studies start at 50000 and 60000, with 2000 episodes per condition. The separate model protocol uses 200 episodes starting at 70000. Primary inference is the eight paired baseline-cost contrasts of adaptive against five fixed policies and three ablations; Holm correction covers all eight. Sensitivity/risk intervals are exploratory and pointwise. Bootstrap and sign-flip seeds are fixed in source.

## Separate model study
No real model was available: fresh discovery found no credentials or running local endpoint; official runtime acquisition had timed out. Evidence is retained in results/llm/status.json and src/llm_study/availability_attempt.json. Mock outcomes test the harness only and are excluded from scientific estimates. Public prompts contain documentation and observations, never hidden outcome truth. The cached client supports OpenAI, Anthropic, local OpenAI-compatible and Ollama interfaces, temperature zero and a persistent USD 5 cap. Paid invocation requires conservative prices explicitly:

```bash
python src/llm_study/run.py --backend openai --model ACTUAL_MODEL --input-price-per-million ACTUAL_PRICE --output-price-per-million ACTUAL_PRICE
```

Record the exact provider/model/version and current verified prices before such an experiment. No API secret is saved or packaged.

## Files and metadata
src/world.py defines timeout-conditioned latent services; policies.py implements read semantics, eventual effect counts and controllers. results contains paired raw episodes and derived tables; figures contains 300-dpi plots. paper has the original template, builder, central author.json, audits and supplementary_p3.zip. literature contains verified primary records and full-text positioning notes. The ZIP has a SHA-256 manifest and includes runnable sources/results/template.

Edit paper/author.json and rebuild with `python paper/build_manuscript.py`; from paper/ use `python build_manuscript.py`. Export the PDF again after metadata changes. Original draft files are backed up outside this isolated project.
