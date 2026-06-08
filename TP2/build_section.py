#!/usr/bin/env python3
import json, re
d = json.load(open("report/analysis_4seed.json"))
M = d["grid_manifest"]
short = {"1159_25": "orange juice", "1159_29": "palm tree", "1159_3": "fire character",
         "1159_7": "hedgehog", "7836": "astronaut", "9338": "rainbow hamster"}
stems = ["1159_25", "1159_29", "1159_3", "1159_7", "7836", "9338"]

def esc(p):
    p = p.replace("‘", "'").replace("’", "'")
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        p = p.replace(a, b)
    return p

def ms(s, nd=3):
    return f"${s['mean']:.{nd}f}\\pm{s['std']:.{nd}f}$"

ac = d["all_candidates"]; bp = d["best_per_image"]; t3 = d["submitted_top3"]; pb = d["per_branch"]
sp = d["spearman"]; ntot = f"{d['n_total']:,}".replace(",", "{,}")

# ---- annex rows --------------------------------------------------------------
annex_rows = []
for st in stems:
    for i, br in enumerate(["clip", "lpips", "mse"]):
        c = M[f"{st}__{br}"]
        tgt = (r"\multirow{3}{*}{\shortstack[l]{" + short[st] + r"\\" + st.replace("_", r"\_") + "}}") if i == 0 else ""
        annex_rows.append(f"{tgt} & {br} & {esc(c['prompt'])} & ${c['clip_similarity']:.3f}$ & ${c['pixel_rmse']:.3f}$ & ${c['lpips']:.3f}$ \\\\")
    annex_rows.append(r"\midrule")
annex_rows[-1] = r"\bottomrule"
annex_body = "\n".join(annex_rows)

# ---- 6x5 grid ----------------------------------------------------------------
def gi(name):
    return r"\includegraphics[width=0.150\textwidth]{figures/grid/" + name + "}"
hdr = " & " + " & ".join(r"\scriptsize " + short[st] for st in stems) + r" \\[1pt]"
def grid_row(label, key):
    cells = " & ".join(gi(f"{st}__{key}.png") for st in stems)
    return r"\rotatebox{90}{\scriptsize " + label + "} & " + cells + r" \\"
grid = "\n".join([
    hdr,
    grid_row("Target", "target"),
    grid_row("CLIP", "clip"),
    grid_row("LPIPS", "lpips"),
    grid_row("MSE", "mse"),
    grid_row("Composite", "composite"),
])

ANALYSIS = r"""\section{Analysis of the Results}
\label{sec:results}

Every number in this section pools the four completed optimiser-seed runs ($1234$, $3456$, $5678$, $9012$), each a full closed-loop search with the LCM render seed held fixed per image. This gives \textbf{""" + ntot + r"""} scored candidates across the six targets. Aggregating over four independent searches also removes the single-run diversity collapse noted in earlier experiments: every target now contributes three genuinely distinct strong prompts, so all $18$ top-3 slots are filled.

\subsection{Aggregate metrics across the test set}
Table~\ref{tab:aggregate} reports the mean$\pm$std of the three metrics at three levels of aggregation: over \emph{all} pooled candidates, over the single \emph{best} candidate per image on each metric (top-1), and over the \emph{submitted top-3}.

\begin{table}[H]
\centering
\caption{Aggregate image-side metrics over the four-seed pool (mean$\pm$std). CLIP higher is better; LPIPS and pixel MSE lower is better.}
\label{tab:aggregate}
\small
\begin{tabular}{l c c c}
\toprule
\textbf{Aggregation} & \textbf{CLIP\,$\uparrow$} & \textbf{LPIPS\,$\downarrow$} & \textbf{Pixel MSE\,$\downarrow$} \\
\midrule
All candidates ($n{=}""" + ntot + r"""$) & """ + ms(ac["clip_similarity"]) + " & " + ms(ac["lpips"]) + " & " + ms(ac["pixel_mse"], 4) + r""" \\
Best per image ($n{=}6$) & """ + r"$\mathbf{" + f"{bp['clip_similarity']['mean']:.3f}\\pm{bp['clip_similarity']['std']:.3f}" + r"}$ & $\mathbf{" + f"{bp['lpips']['mean']:.3f}\\pm{bp['lpips']['std']:.3f}" + r"}$ & $\mathbf{" + f"{bp['pixel_mse']['mean']:.4f}\\pm{bp['pixel_mse']['std']:.4f}" + r"}$ \\" + r"""
Submitted top-3 ($n{=}18$) & """ + ms(t3["clip_similarity"]) + " & " + ms(t3["lpips"]) + " & " + ms(t3["pixel_mse"], 4) + r""" \\
\bottomrule
\end{tabular}
\end{table}

\noindent
The best reconstruction per image is strong on all three metrics at once (CLIP """ + f"{bp['clip_similarity']['mean']:.3f}" + r""", LPIPS """ + f"{bp['lpips']['mean']:.3f}" + r""", pixel MSE """ + f"{bp['pixel_mse']['mean']:.4f}" + r"""), which is the payoff of the closed loop: by rendering and scoring at every iteration, the search fixes the specific error of the current best candidate instead of hoping a blind sample lands well. The full pool is much wider (CLIP std """ + f"{ac['clip_similarity']['std']:.3f}" + r""") because it includes every exploratory proposal across four seeds and twelve iterations, many of them deliberately off-distribution. The submitted top-3 sits close to the best-per-image row, confirming that pooling four seeds supplies three strong, distinct prompts per target rather than one good prompt and two weak echoes.

\subsection{Best candidate per branch}
Each branch is steered by a single objective, so the fairest way to read it is on its own metric. For every (target, seed) pair we take that branch's best candidate under its own objective, then average over the six targets and four seeds ($n{=}24$ per branch). Table~\ref{tab:perbranch} shows the expected diagonal dominance: each branch leads the field on the metric it optimises (\textbf{bold}), while the composite branch, selected by the scale-free rank-average, is never the outright winner on any single metric but is a close second everywhere, which is exactly the balance we want from the final ranker.

\begin{table}[H]
\centering
\caption{Best candidate per branch, evaluated on the branch's own objective and averaged over six targets $\times$ four seeds ($n{=}24$). Bold marks the metric each branch optimises. The composite branch is selected by the rank-average and shown for comparison.}
\label{tab:perbranch}
\small
\begin{tabular}{l c c c}
\toprule
\textbf{Branch} & \textbf{CLIP\,$\uparrow$} & \textbf{LPIPS\,$\downarrow$} & \textbf{RMSE\,$\downarrow$} \\
\midrule
CLIP & $\mathbf{""" + f"{pb['clip']['clip_similarity']['mean']:.3f}\\pm{pb['clip']['clip_similarity']['std']:.3f}" + r"}$ & " + ms(pb['clip']['lpips']) + " & " + ms(pb['clip']['pixel_rmse']) + r""" \\
LPIPS & """ + ms(pb['lpips']['clip_similarity']) + r" & $\mathbf{" + f"{pb['lpips']['lpips']['mean']:.3f}\\pm{pb['lpips']['lpips']['std']:.3f}" + r"}$ & " + ms(pb['lpips']['pixel_rmse']) + r""" \\
MSE & """ + ms(pb['mse']['clip_similarity']) + " & " + ms(pb['mse']['lpips']) + r" & $\mathbf{" + f"{pb['mse']['pixel_rmse']['mean']:.3f}\\pm{pb['mse']['pixel_rmse']['std']:.3f}" + r"}$ \\" + r"""
Composite & """ + ms(pb['composite']['clip_similarity']) + " & " + ms(pb['composite']['lpips']) + " & " + ms(pb['composite']['pixel_rmse']) + r""" \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Best reconstruction per metric}
Figure~\ref{fig:grid} shows, for each target (columns) and each objective (rows), the single best render selected across all four seeds, with the target row on top for reference. Reading down a column exposes how the objectives disagree: the CLIP-best render is usually the most semantically faithful, the MSE-best render the closest in raw pixels (often by matching the background), and the LPIPS-best somewhere between. The prompts behind the CLIP, LPIPS and MSE rows are the submitted top-3 and are listed in full in Annex~\ref{annex:top3}.

\begin{figure}[H]
\centering
\setlength{\tabcolsep}{1.2pt}
\renewcommand{\arraystretch}{0.4}
\begin{tabular}{c@{\hskip 2pt}cccccc}
""" + grid + r"""
\end{tabular}
\caption{Best reconstruction per (target, objective) over the four-seed pool. Columns are the six targets; rows are the target image and the CLIP-, LPIPS-, MSE- and composite-optimal renders. All renders use the fixed LCM configuration with the seed from the filename.}
\label{fig:grid}
\end{figure}

\subsection{Do the metrics agree? Rank correlations}
Table~\ref{tab:spearman} reports Spearman rank correlations between the three metrics over the whole pooled candidate set, and Figure~\ref{fig:corr} visualises the two most informative pairings.

\begin{table}[H]
\centering
\caption{Spearman rank correlation between metrics over the pooled candidate set ($n{=}""" + ntot + r"""$).}
\label{tab:spearman}
\small
\begin{tabular}{c c c}
\toprule
\textbf{CLIP vs LPIPS} & \textbf{CLIP vs MSE} & \textbf{LPIPS vs MSE} \\
\midrule
$""" + f"{sp['clip_vs_lpips']:+.3f}" + r"$ & $" + f"{sp['clip_vs_mse']:+.3f}" + r"$ & $" + f"{sp['lpips_vs_mse']:+.3f}" + r"""$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.86\textwidth]{figures/corr_scatter.png}
\caption{Each point is one of the """ + ntot + r""" pooled candidates. CLIP is nearly orthogonal to the low-level metrics (left, mildly negative), whereas LPIPS and pixel MSE are strongly positively related (right).}
\label{fig:corr}
\end{figure}

\noindent
Two facts stand out. First, \textbf{CLIP similarity is essentially decoupled from the pixel/perceptual metrics}: it is mildly anti-correlated with LPIPS ($""" + f"{sp['clip_vs_lpips']:+.3f}" + r"""$) and effectively uncorrelated with pixel MSE ($""" + f"{sp['clip_vs_mse']:+.3f}" + r"""$). A prompt can nail the target's semantics (high CLIP) while differing in layout and texture, so optimising CLIP alone would not minimise pixel error. Second, the two low-level metrics agree strongly ($""" + f"{sp['lpips_vs_mse']:+.3f}" + r"""$): a candidate close in raw pixels is usually close perceptually too. Together this justifies keeping the objectives separate and combining them only through the scale-free composite ranking, rather than collapsing everything into one number.

\subsection{Stability across seeds}
Figure~\ref{fig:seedstab} reports the best-per-image mean of each metric within each of the four seeds. The bars barely move (best CLIP stays in $0.885$--$0.893$, best LPIPS in $0.471$--$0.498$, best MSE in $0.018$--$0.020$), so no single lucky seed drives the pooled statistics; the four searches converge to comparable quality, and pooling mainly buys prompt \emph{diversity} rather than higher peak fidelity.

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/seed_stability.png}
\caption{Best-per-image metric means within each optimiser seed (error bars are std over the six targets). The four seeds are mutually consistent.}
\label{fig:seedstab}
\end{figure}

\subsection{Qualitative analysis: successes, failures, ambiguous cases}
\label{sec:limits}
\paragraph{\normalfont\textbf{Clear success, orange juice (1159\_25).}} The easiest target. Short, literal prompts reach CLIP up to $0.963$ and the lowest LPIPS of the set ($0.347$); the MSE-best render is closest in raw pixels (RMSE $0.129$). The subject is a single common photographable object with no stylistic ambiguity, so warm start and refinement converge quickly and all three objectives land on near-identical, faithful images.

\paragraph{\normalfont\textbf{High CLIP, high LPIPS, palm tree (1159\_29).}} The semantics are recovered very well (CLIP up to $0.935$) but LPIPS stays high (best only $0.606$) and MSE is among the worst of the set. The scene is correct in content (a palm tree on a rock in the ocean at sunset) but its exact composition (horizon line, wave pattern, sun position) is hard to pin down from text, so perceptual distance stays large despite strong semantic agreement. This is the clearest case of the metric disagreement quantified in Table~\ref{tab:spearman}.

\paragraph{\normalfont\textbf{Hardest semantic target, hedgehog (1159\_7).}} The lowest CLIP of the set: even the CLIP-best prompt reaches only $0.808$. The target is an out-of-distribution concept (a hedgehog of crystalline/translucent spikes on a foam cube) that the captioners struggle to name and the generator cannot reliably reproduce. This target also produced the clearest example of a \emph{gamed} metric: the LPIPS-best render came from the near-empty prompt ``Hedgehog, light amber coat, teen'' (CLIP only $0.529$), which scores well on LPIPS by being bland and low-detail while losing the subject entirely. It is a concrete argument against trusting any single low-level metric, and for keeping the composite ranker for the final selection.

\paragraph{\normalfont\textbf{Lowest absolute error, astronaut (7836).}} The lowest pixel error of the set (RMSE $0.101$, LPIPS down to $0.391$) with CLIP up to $0.917$, driven by the large dark, low-frequency background that is trivial to reproduce, a reminder that pixel MSE rewards background agreement as much as subject fidelity.

\paragraph{\normalfont\textbf{Diversity recovered by multi-seed pooling.}} In single-seed runs the aggressive image-level de-duplication collapsed easy targets (the rainbow hamster, the astronaut) onto near-identical prompts, so the submitted top-3 occasionally fell short of three distinct candidates. Pooling four independent optimiser seeds resolves this: because each seed explores a different sampling trajectory, the union supplies three genuinely distinct strong prompts for every target and all $18$ slots are filled (Annex~\ref{annex:top3}). The residual limitation is no longer diversity but metric gaming, as the hedgehog case shows, which is mitigated but not eliminated by the composite ranking."""

ANNEX = r"""

\clearpage
\section*{Annex A: Full submitted top-3 prompts}
\addcontentsline{toc}{section}{Annex A: Full submitted top-3 prompts}
\label{annex:top3}
For each target, the three submitted prompts are the CLIP-, LPIPS- and MSE-best renders of Figure~\ref{fig:grid} (the composite render is excluded, as the composite is the cross-metric ranker rather than a fourth submission). Prompts are reproduced verbatim, including their original phrasing. Metrics are the values of that exact render under the fixed LCM configuration.

\begin{table}[H]
\centering
\caption{Full submitted top-3 prompts per target, with image-side metrics. ``Selected by'' is the objective the render is best under.}
\label{tab:annex-top3}
\scriptsize
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{p{1.5cm} l p{5.6cm} c c c}
\toprule
\textbf{Target} & \textbf{Sel.\ by} & \textbf{Prompt (verbatim)} & \textbf{CLIP\,$\uparrow$} & \textbf{RMSE\,$\downarrow$} & \textbf{LPIPS\,$\downarrow$} \\
\midrule
""" + annex_body + r"""
\end{tabular}
\end{table}
"""

# ---- splice ------------------------------------------------------------------
src = open("report/main.tex").read()
start = src.index(r"\section{Analysis of the Results}")
end = src.index(r"\section{Conclusion}")
src = src[:start] + ANALYSIS + "\n\n" + src[end:]
src = src.replace(r"\end{document}", ANNEX + "\n\\end{document}")
open("report/main.tex", "w").write(src)
print("spliced. analysis chars:", len(ANALYSIS), "annex rows:", len(annex_rows))
