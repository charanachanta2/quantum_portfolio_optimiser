# One command per stage (Linux/macOS/WSL).  Windows: run the python commands directly.
PY=python
data:        ; $(PY) scripts/fetch_data.py && $(PY) scripts/build_esg.py
qec:         ; $(PY) scripts/run_all.py --stage qec --out results_real
resources:   ; $(PY) scripts/run_all.py --stage resources --out results_real
tune:        ; $(PY) scripts/tune.py --real --target GFC_2008 --out results_real && $(PY) scripts/tune.py --real --target COVID_2020 --out results_real
backtest:    ; $(PY) scripts/run_all.py --real --require-optional --stage backtest --out results_real
ablation:    ; $(PY) scripts/run_all.py --real --stage ablation --out results_real
scaling:     ; $(PY) scripts/run_all.py --real --stage scaling --out results_real
summary:     ; $(PY) scripts/run_all.py --real --stage summary --out results_real
shots:       ; $(PY) scripts/screenshots.py --real --out results_real
report:      ; $(PY) report/build_report.py --results results_real --out report/QPO_Technical_Report.pdf
walkthrough: ; cd ../walkthrough && QPO_RESULTS=../qpo_project/results_real $(PY) figs.py && QPO_RESULTS=../qpo_project/results_real $(PY) content.py && $(PY) mkpdf.py && node mkppt.js
test:        ; $(PY) -m pytest -q
smoke:       ; $(PY) scripts/run_all.py --fast --out results_smoke
all-real: data qec resources tune backtest ablation scaling summary shots report walkthrough

# ---- Part 2: combined system -------------------------------------------------
train-predictor:        ## offline LSTM training (deployed model + walk-forward models for the backtests)
	python scripts/train_predictor.py --hparams models/ga_best.json --epochs 10
	python scripts/train_predictor.py --walk-forward 2007-12-31 2008-12-31 2018-12-31 2019-12-31
ga-tune:
	python scripts/ga_tune.py --train-end 2018-12-31 --pop 6 --gens 3
backtest-lstm:          ## F4 (resumable: re-run to continue)
	python scripts/backtest_lstm.py --real --out results_lstm
app:
	streamlit run app.py
