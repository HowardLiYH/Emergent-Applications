# Expert Panel Discussion: Agents & Title

**Date:** January 18, 2026
**Topics:**
1. Are these LLM agents or algorithmic populations?
2. Should the title be more finance-relevant?

---

## Question 1: What Are These "Agents"?

### Code Reality Check

```python
@dataclass
class Agent:
    """Trading agent with niche affinity tracking."""
    agent_id: int
    strategy_idx: int
    niche_affinity: np.ndarray  # Just a probability vector!

    def update_affinity(self, regime_idx: int, won: bool, alpha: float = 0.1):
        """Update based on win/loss - simple exponential update."""
        if won:
            self.niche_affinity[regime_idx] += alpha * (1 - self.niche_affinity[regime_idx])
        else:
            self.niche_affinity[regime_idx] *= (1 - alpha)
```

### Expert Panel Analysis

**Prof. Michael Kearns (Penn, Game Theory):**
> "These are NOT LLM agents. They are:
> - **No neural networks**
> - **No language models**
> - **No learning from data**
> Just simple **affinity vectors** updated via multiplicative weights.
>
> This is closer to **evolutionary algorithms** or **replicator dynamics** than 'agents' in the modern AI sense."

**Dr. Ilya Sutskever (OpenAI):**
> "Calling these 'agents' may confuse readers who expect LLM-based systems. In 2025, 'agent' implies reasoning, planning, tool use. These are **strategy populations**."

**Prof. Sanjeev Arora (Princeton):**
> "The mechanism is beautiful precisely because it's simple - just fitness-proportional updates. But the term 'agent' is overloaded. Consider:
> - 'Strategy populations'
> - 'Competing strategies'
> - 'Replicator units'"

**Quant Director, Two Sigma:**
> "In finance, we'd call these **strategy instances** or **portfolio variants**, not agents. 'Agent' implies autonomy and decision-making complexity."

---

### Panel Vote: What to Call Them?

| Term | Votes | Pros | Cons |
|------|-------|------|------|
| **Agents** | 8/40 | Familiar in MAS literature | Confusing in LLM era |
| **Strategies** | 18/40 | Accurate, finance-friendly | Less exciting |
| **Competing strategies** | 12/40 | Descriptive | Wordy |
| **Replicator units** | 2/40 | Theoretically precise | Too jargon-y |

**Consensus:** Use **"competing strategies"** or **"strategy instances"** in finance contexts. Keep "agents" only when connecting to evolutionary game theory literature.

---

## Question 2: Should Title Be More Finance-Relevant?

### Current Title
> "Emergent Specialization from Competition Alone: How Replicator Dynamics Create Market-Correlated Behavior"

### Expert Panel Discussion

**Prof. Andrew Lo (MIT, Finance):**
> "The current title is good for NeurIPS (AI/ML audience). But if targeting finance journals (JFE, RFS), consider:
> - Adding 'Trading Strategies' or 'Financial Markets'
> - Making 'Specialization Index' more prominent"

**Prof. Yann LeCun (NYU):**
> "Keep it general! The result is about **emergence**, not just finance. Finance is just the testbed. If you make it too finance-specific, it loses the broader AI/complexity science appeal."

**Research Lead, Citadel:**
> "For practitioners, 'Market-Correlated Behavior' is vague. What market correlation? Suggest:
> - 'Trading Strategy Specialization Tracks Market Structure'
> - More specific about ADX/trend strength"

**Prof. David Silver (DeepMind):**
> "The 'Emergent Specialization' framing is strong. It's the hook that makes this interesting beyond finance. Don't lose that."

---

### Title Options

| Option | Target Audience | Panel Votes |
|--------|-----------------|-------------|
| **A. Current:** "Emergent Specialization from Competition Alone: How Replicator Dynamics Create Market-Correlated Behavior" | NeurIPS (AI/ML) | **22/40** |
| **B. Finance-focused:** "How Competing Trading Strategies Develop Market-Synchronized Specialization" | Finance journals | 8/40 |
| **C. Hybrid:** "Emergent Specialization in Trading Strategy Populations: Cointegration with Market Structure" | Both | 10/40 |
| **D. Shortened:** "Emergent Specialization from Competition Alone" | NeurIPS (punchy) | 0/40 |

---

### Recommendation by Venue

| Venue | Recommended Title |
|-------|-------------------|
| **NeurIPS** | Option A (current) - emphasizes emergence, replicator dynamics |
| **ICML** | Option A (current) |
| **AAAI** | Option A (current) |
| **Journal of Finance** | Option B - finance-focused |
| **Quantitative Finance** | Option C - hybrid |
| **Risk Magazine** | Option B - practitioner-friendly |

---

## Final Recommendations

### 1. Agent Terminology
```
RECOMMENDED CHANGES:

Abstract: "agents" → "competing strategies"
Paper body: Keep "agents" when discussing MAS literature
           Use "strategy instances" when discussing implementation
Algorithm 1: Keep "Agent" as dataclass name (code convention)
README: Clarify that these are NOT LLM agents
```

### 2. Title Decision

**For NeurIPS submission:** Keep current title (Option A)
- "Emergent Specialization from Competition Alone: How Replicator Dynamics Create Market-Correlated Behavior"

**Rationale:**
- Emphasizes the **mechanism** (emergence, replicator dynamics)
- Appeals to AI/ML audience
- Finance is testbed, not the main contribution

**If rejected and resubmitting to finance venue:** Switch to Option C
- "Emergent Specialization in Trading Strategy Populations: Cointegration with Market Structure"

---

## Panel Consensus

| Decision | Votes | Action |
|----------|-------|--------|
| Clarify "agents" are not LLMs | **38/40** | Add sentence in paper |
| Keep current title for NeurIPS | **32/40** | No change |
| Update README to clarify | **40/40** | Implement |

*Discussion complete*
