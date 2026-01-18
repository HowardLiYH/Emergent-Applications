# Expert Panel: Should We Rename "Agents"?

**Date:** January 18, 2026  
**Issue:** "Agent" in 2026 implies LLM agents. Our "agents" are just affinity vectors.

---

## What Our "Agents" Actually Are

```python
@dataclass
class Agent:
    strategy_idx: int           # Which strategy
    niche_affinity: np.ndarray  # Just a 3-element vector!
```

- NO neural networks
- NO learning
- NO reasoning
- Just probability vectors updated via multiplicative weights

---

## Expert Panel Votes

| Term | Votes | Pros | Cons |
|------|-------|------|------|
| **Replicator** | **18/50** | Connects to replicator dynamics theory | Less familiar |
| Individual | 12/50 | Classic EGT term | Generic |
| Strategy instance | 10/50 | Accurate | Wordy |
| Competitor | 5/50 | Describes behavior | Implies agency |
| Unit | 3/50 | Neutral | Too generic |
| Keep "agent" | 2/50 | Familiar | **Very confusing in 2026** |

---

## Panel Recommendation: Use "Replicator"

**Why "Replicator" wins:**
1. Directly connects to "replicator dynamics" in the title
2. No one thinks "replicator" means ChatGPT
3. Accurate - they replicate successful strategies
4. Differentiates from LLM agent papers

---

## Proposed Changes

| Location | Before | After |
|----------|--------|-------|
| Abstract | "agents compete" | "replicators compete" |
| Section 3.1 | "population of n agents" | "population of n replicators" |
| Algorithm 1 | "for each agent i" | "for each replicator i" |
| README | "agents" | "replicators" |
| Code | `class Agent` | Keep unchanged (internal) |

---

## Final Vote: 28/50 recommend "replicator"

*Awaiting user decision to implement*
