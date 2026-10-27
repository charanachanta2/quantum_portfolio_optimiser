"""Download the reference papers from arXiv and merge them into ONE pdf with bookmarks.
Usage (PowerShell):  pip install pypdf requests
                     python merge_papers.py
Optional: put QPO_Results_and_Architecture.pdf in the same folder -> it becomes the first section.
Any paper that fails to download is listed at the end; download it by hand into ./papers/<id>.pdf and re-run."""
import os, sys, time, requests
from pypdf import PdfWriter, PdfReader

PAPERS = [  # (arXiv id, short title)
 ("2410.16265", "Yuan et al. - Quantifying the advantages of QAOA for portfolio optimisation"),
 ("2603.16904", "Weinberg - Walk-forward QUBO scheduling via QAOA"),
 ("2607.03218", "Quantum annealing for dynamic portfolio optimization under transaction costs"),
 ("2509.17876", "Stopfer & Wagner - Quantum portfolio optimization: an extensive benchmark"),
 ("2207.10555", "Brandhofer et al. - Benchmarking portfolio optimization with QAOA"),
 ("2009.05532", "Stilck Franca & Garcia-Patron - Limitations of optimization on noisy devices"),
 ("1612.02058", "Temme, Bravyi, Gambetta - Error mitigation for short-depth circuits"),
 ("2103.02202", "Gidney - Stim"),
 ("2303.15933", "Higgott & Gidney - Sparse Blossom (PyMatching v2)"),
 ("1403.2975",  "Ross & Selinger - Optimal ancilla-free Clifford+T z-rotations"),
 ("1411.4028",  "Farhi et al. - QAOA"),
 ("1811.03975", "Rebentrost & Lloyd - Quantum computational finance"),
 ("1808.06709", "Fowler & Gidney - Lattice surgery"),
 ("2201.02773", "Herman et al. - Quantum computing for finance"),
]
os.makedirs("papers", exist_ok=True)
w = PdfWriter(); failed = []
front = "QPO_Results_and_Architecture.pdf"
if os.path.exists(front):
    n = len(w.pages); w.append(front); w.add_outline_item("QPO results and architecture", n)
for aid, title in PAPERS:
    path = f"papers/{aid}.pdf"
    if not os.path.exists(path):
        try:
            r = requests.get(f"https://arxiv.org/pdf/{aid}", timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"): raise ValueError("not a pdf")
            open(path, "wb").write(r.content); time.sleep(3)      # be polite to arXiv
        except Exception as e:
            failed.append((aid, str(e))); continue
    try:
        n = len(w.pages); w.append(PdfReader(path)); w.add_outline_item(f"arXiv:{aid}  {title}", n)
    except Exception as e:
        failed.append((aid, f"unreadable: {e}"))
w.write("QPO_Results_plus_References.pdf")
print("wrote QPO_Results_plus_References.pdf,", len(w.pages), "pages")
for a, e in failed: print("FAILED", a, e)
