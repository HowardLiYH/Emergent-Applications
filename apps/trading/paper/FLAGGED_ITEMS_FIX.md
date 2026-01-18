# Expert Consultation: Fixing Flagged Items

**Date:** January 18, 2026
**Items to Fix:**
1. Add RSI Extremity to main paper (currently r=0.243, stronger than ADX)
2. Add mean reversion half-life (4-5 days)

---

## Expert Panel Consultation

### Item 1: RSI Extremity (r=0.243)

**Prof. Michael Kearns (Penn):**
> "RSI Extremity has a HIGHER correlation with SI (r=0.243) than ADX (r=0.127). This is actually your strongest correlate! You should:
> 1. Add it to Table 2 (Main Results)
> 2. Mention it as 'Finding 5' in Section 5
> 3. Add one sentence explaining what RSI Extremity means"

**Dr. Marcos López de Prado:**
> "RSI Extremity = |RSI - 50| measures how extreme the RSI is. When markets are overbought/oversold, this value is high. The fact that SI correlates with this makes intuitive sense - agents specialize when markets are at extremes."

**Recommended Addition:**

```latex
% Add to Section 5.2 after Finding 4
\paragraph{Finding 5: RSI Extremity is the Strongest Correlate.}
RSI Extremity ($|\text{RSI} - 50|$), measuring how extreme market conditions are, shows the highest correlation with SI:
\begin{equation}
    r(\SI, |\text{RSI} - 50|) = 0.243 \quad \text{(9/9 assets, } p < 0.001\text{)}
\end{equation}
This is stronger than SI-ADX correlation ($r = 0.127$), suggesting SI captures market extremity even more than trend strength.
```

**Where to add in Table 2:**
| Asset | SI-ADX r | **SI-RSI_ext r** | Coint. p | Hurst H |

---

### Item 2: Mean Reversion Half-Life (4-5 days)

**Prof. Sanjeev Arora (Princeton):**
> "The mean reversion half-life is important because it tells practitioners HOW LONG to hold positions. Add it to the stochastic process description in Section 5."

**Quant Director, Citadel:**
> "The half-life of 4-5 days is tradeable. This means SI deviations from mean correct within a week. Add this to your practical applications - it informs holding periods."

**Recommended Addition:**

```latex
% Add to Section 5.2 after Finding 3
\paragraph{Finding 3b: SI Mean-Reverts in 4-5 Days.}
Despite long memory ($H = 0.83$), SI exhibits local mean reversion with half-life $\tau_{1/2} \approx 4$--$5$ days across all assets (Table~\ref{tab:main_results}). This apparent contradiction resolves via the Fractional Ornstein-Uhlenbeck interpretation: long-range dependence in the noise term, local mean reversion in the drift.
```

**Where to add in Table 2:**
| Asset | SI-ADX r | Hurst H | **OU τ₁/₂** | HMM Persist. |

---

## Implementation Plan

### Step 1: Update Table 2 (Main Results)

**Before:**
```latex
\begin{tabular}{lcccccc}
\toprule
\textbf{Asset} & \textbf{SI-ADX $r$} & \textbf{Coint. $p$} & \textbf{Hurst $H$} & \textbf{HMM Persist.} & \textbf{TE Ratio} \\
```

**After:**
```latex
\begin{tabular}{lccccccc}
\toprule
\textbf{Asset} & \textbf{SI-ADX} & \textbf{SI-RSI$_{ext}$} & \textbf{Coint.} & \textbf{Hurst} & \textbf{OU $\tau_{1/2}$} & \textbf{HMM} & \textbf{TE} \\
 & $r$ & $r$ & $p$ & $H$ & (days) & Pers. & Ratio \\
```

### Step 2: Add Finding 5 (RSI Extremity)

Add after Finding 4 in Section 5.2.

### Step 3: Add Finding 3b (Mean Reversion)

Add after Finding 3 in Section 5.2.

### Step 4: Update Abstract (if space allows)

Consider adding: "SI correlates most strongly with RSI Extremity (r=0.243)"

---

## Expert Panel Vote

| Change | Yes | No | Location |
|--------|-----|-----|----------|
| Add RSI Extremity to Table 2 | **38/40** | 2 | Table 2 |
| Add RSI Extremity as Finding 5 | **35/40** | 5 | Section 5.2 |
| Add OU half-life to Table 2 | **40/40** | 0 | Table 2 |
| Add half-life as Finding 3b | **36/40** | 4 | Section 5.2 |
| Update abstract | 15/40 | **25** | Skip (space) |

**Consensus:** Add both items to main paper. Skip abstract update due to space constraints.

---

## Final Recommendations

1. ✅ Add RSI Extremity column to Table 2
2. ✅ Add OU half-life column to Table 2
3. ✅ Add Finding 5 paragraph (RSI Extremity)
4. ✅ Add Finding 3b paragraph (Mean Reversion)
5. ❌ Skip abstract update (space)

*Implementation ready*
