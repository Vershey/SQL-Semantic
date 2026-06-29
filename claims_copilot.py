#!/usr/bin/env python3
"""
Claims Coverage Copilot
=======================
An *advanced-RAG* demo for the insurance domain that you drive with prompts.

Agentic pipeline (each step is a real piece of an advanced-RAG system):

    1. plan      -> an LLM planner decides which sources/queries to run and
                    which real-time tools to call
    2. retrieve  -> HYBRID lexical search: BM25 + TF-IDF, fused with
                    Reciprocal Rank Fusion (RRF)
    3. rerank    -> an LLM reranker re-scores the candidates (the "semantic"
                    quality layer; swap in a cross-encoder in production)
    4. tools     -> optional REAL-TIME signals: weather/CAT + sanctions screen
                    (stubbed here; marked where you wire in a live API)
    5. draft     -> an LLM drafts a coverage determination WITH citations
    6. verify    -> an LLM verifier checks every claim against the retrieved
                    evidence and ABSTAINS instead of guessing when unsupported

Run it two ways
---------------
    # Fully offline, no key needed (canned LLM responses) -- proves the flow:
    python claims_copilot.py --dry-run

    # Real Anthropic API (set your key first, never paste it in code/chat):
    export ANTHROPIC_API_KEY="sk-ant-..."     # macOS / Linux
    setx   ANTHROPIC_API_KEY "sk-ant-..."     # Windows (new shell after)
    python claims_copilot.py

Docs for the API: https://docs.claude.com/en/api/overview
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

import numpy as np
from rank_bm25 import BM25Okapi

# --------------------------------------------------------------------------- #
# Config — models are configurable via env vars. These are current API strings.
# --------------------------------------------------------------------------- #
REASONING_MODEL = os.environ.get("COPILOT_MODEL", "claude-sonnet-4-6")
RERANK_MODEL = os.environ.get("COPILOT_RERANK_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 1200
TOP_K_RETRIEVE = 6   # candidates after hybrid retrieval
TOP_N_EVIDENCE = 4   # docs kept after rerank and passed to the drafter

# --------------------------------------------------------------------------- #
# Synthetic insurance corpus.
# In production these chunks come from your policy admin system, claims DB,
# and guideline store. NEVER load real customer PII/PHI into a demo.
# --------------------------------------------------------------------------- #
CORPUS: list[dict] = [
    {
        "id": "POL-BASE",
        "source": "Policy HO-1002, Section I — Coverages",
        "dtype": "policy",
        "text": (
            "We insure the dwelling and personal property against sudden and "
            "accidental direct physical loss, including water that escapes from "
            "a plumbing, heating, or air-conditioning system, or from a household "
            "appliance, when the discharge is sudden and accidental."
        ),
    },
    {
        "id": "END-HO217",
        "source": "Endorsement HO-217 — Water Exclusion",
        "dtype": "endorsement",
        "text": (
            "This endorsement modifies Policy HO-1002. We do not cover loss "
            "caused directly or indirectly by flood, surface water, waves, "
            "tidal water, overflow of a body of water, or spray from any of "
            "these, whether or not driven by wind. Such loss is excluded "
            "regardless of any other cause contributing concurrently."
        ),
    },
    {
        "id": "END-HO305",
        "source": "Endorsement HO-305 — Water Backup and Sump Overflow",
        "dtype": "endorsement",
        "text": (
            "This endorsement adds limited coverage to Policy HO-1002. We cover "
            "direct physical loss caused by water that backs up through sewers "
            "or drains, or overflows from a sump pump, subject to a sublimit of "
            "$5,000 per occurrence. This coverage does not apply to flood or "
            "surface water as defined in the policy."
        ),
    },
    {
        "id": "DEF-FLOOD",
        "source": "Policy HO-1002, Definitions",
        "dtype": "policy",
        "text": (
            "Flood and surface water mean water on the surface of the ground, "
            "including water that accumulates from rainfall and runs off or "
            "ponds before entering a drain or sewer system."
        ),
    },
    {
        "id": "GUIDE-WTR",
        "source": "Claims Handling Guideline — Water Losses",
        "dtype": "guideline",
        "text": (
            "Adjusters must distinguish sudden internal discharge (generally "
            "covered) from flood or surface water (excluded unless a flood "
            "endorsement applies). Where heavy rainfall is involved, pull weather "
            "and catastrophe data for the loss date and location, and determine "
            "the proximate cause of the intrusion before deciding coverage."
        ),
    },
    {
        "id": "CLM-2201",
        "source": "Prior claim 2201 (similar fact pattern)",
        "dtype": "claim",
        "text": (
            "Basement water intrusion after heavy rain. Investigation found "
            "surface water entered through a foundation crack. Claim denied "
            "under the water exclusion endorsement; no backup coverage applied."
        ),
    },
    {
        "id": "CLM-2202",
        "source": "Prior claim 2202 (contrast fact pattern)",
        "dtype": "claim",
        "text": (
            "Water heater tank ruptured and discharged into the basement. "
            "Determined to be sudden and accidental discharge from an appliance; "
            "claim covered for dwelling and personal property."
        ),
    },
]
ID_TO_DOC = {d["id"]: d for d in CORPUS}

DEFAULT_CLAIM = (
    "FNOL: Homeowner reports water in the finished basement after heavy rain "
    "on 2026-03-14 in Austin, TX. Standing water ~3 inches; damage to drywall "
    "and personal property. Claimant: John Roe. Policy: HO-1002. Cause not yet "
    "confirmed — possibly surface water, possibly a drain backup."
)


# --------------------------------------------------------------------------- #
# Hybrid retriever: BM25 + TF-IDF, fused with Reciprocal Rank Fusion.
# --------------------------------------------------------------------------- #
def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """Lexical hybrid (BM25 + TF-IDF) with RRF fusion.

    This runs anywhere with no API key or network. To make it a true
    lexical+semantic hybrid, add an EmbeddingRetriever (Voyage / OpenAI /
    sentence-transformers) as a third ranked list and fuse it in below.
    """

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.ids = [d["id"] for d in corpus]
        toks = [_tok(d["text"] + " " + d["source"]) for d in corpus]
        self.bm25 = BM25Okapi(toks)
        # --- tiny TF-IDF in numpy (no sklearn needed) ---
        self.vocab = sorted({t for doc in toks for t in doc})
        self.vindex = {t: i for i, t in enumerate(self.vocab)}
        n_docs, n_vocab = len(corpus), len(self.vocab)
        tf = np.zeros((n_docs, n_vocab))
        for di, doc in enumerate(toks):
            for t in doc:
                tf[di, self.vindex[t]] += 1
        df = (tf > 0).sum(axis=0)
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        mat = tf * self.idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.doc_vecs = mat / np.clip(norms, 1e-9, None)

    def _tfidf_query_vec(self, query: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in _tok(query):
            if t in self.vindex:
                v[self.vindex[t]] += 1
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n else v

    @staticmethod
    def _ranks(scores: np.ndarray) -> dict[int, int]:
        order = np.argsort(-scores)
        return {int(idx): rank for rank, idx in enumerate(order)}

    def search(self, queries: list[str], top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        rrf, k = {}, 60  # standard RRF constant
        for q in queries:
            bm_ranks = self._ranks(np.array(self.bm25.get_scores(_tok(q))))
            tf_ranks = self._ranks(self.doc_vecs @ self._tfidf_query_vec(q))
            for ranks in (bm_ranks, tf_ranks):
                for di, r in ranks.items():
                    rrf[di] = rrf.get(di, 0.0) + 1.0 / (k + r)
        top = sorted(rrf, key=lambda di: -rrf[di])[:top_k]
        return [self.corpus[di] for di in top]


# --------------------------------------------------------------------------- #
# Real-time tools (STUBBED). Swap the bodies for live API calls.
# The container/network must allow egress to the provider's domain.
# --------------------------------------------------------------------------- #
def tool_weather_cat(date: str, location: str) -> dict:
    # TODO: replace with a live call, e.g. National Weather Service / NOAA.
    return {
        "tool": "weather_cat",
        "date": date,
        "location": location,
        "event": "Flash Flood Warning in effect",
        "severity": "moderate",
        "source": "NWS (stubbed)",
    }


def tool_sanctions_screen(name: str) -> dict:
    # TODO: replace with a live screen against the public OFAC SDN list.
    return {"tool": "sanctions_screen", "name": name, "match": False, "list": "OFAC SDN (stubbed)"}


TOOLS = {"weather_cat": tool_weather_cat, "sanctions_screen": tool_sanctions_screen}


# --------------------------------------------------------------------------- #
# Prompt templates. System prompts are constants; user prompts are built from
# data at call time. These four prompts ARE the product — tune them here.
# --------------------------------------------------------------------------- #
PLANNER_SYSTEM = """You are the retrieval planner for an insurance claims copilot.
Given a claim (FNOL), decide what to retrieve and which real-time tools to call.
Available tools: "weather_cat" (weather/catastrophe by date+location),
"sanctions_screen" (screen a named party).
Respond with ONLY a JSON object, no prose, in this schema:
{
  "queries": ["3-5 short retrieval queries covering coverage, exclusions, prior claims"],
  "tools": ["subset of available tool names that are relevant"],
  "reasoning": "one sentence"
}"""

RERANK_SYSTEM = """You are a relevance reranker for insurance retrieval.
Given the claim and a list of candidate documents (id + text), order the ids
from most to least relevant for deciding coverage.
Respond with ONLY JSON: {"ranked_ids": ["ID", "ID", ...]}"""

DRAFTER_SYSTEM = """You are an insurance coverage analyst drafting a FIRST-DRAFT
coverage determination for a human adjuster to review. Use ONLY the supplied
evidence and tool results. Every point in the rationale must trace to a cited
doc_id. If the cause of loss is not established by the evidence, say so and
prefer PARTIAL or flag it as an open question rather than guessing.
Also assign a confidence score (0-100) reflecting how certain the decision is:
  100 = cause of loss unambiguous, policy language clear, no open questions
    0 = cause entirely unknown, conflicting exclusions, evidence absent
Respond with ONLY JSON in this schema:
{
  "decision": "COVERED | PARTIAL | DENIED",
  "confidence": <integer 0-100>,
  "rationale": "2-4 sentences, each grounded in a cited doc_id",
  "citations": [{"doc_id": "ID", "point": "what this doc establishes"}],
  "open_questions": ["facts an adjuster must still confirm"]
}"""

VERIFIER_SYSTEM = """You are a faithfulness verifier. Check the draft determination
against the evidence. Confirm every cited point is actually supported by the
referenced doc. If any material point is unsupported, set all_supported=false and
set final_decision to "ABSTAIN" (escalate to a human).
Also assign a confidence score (0-100) for the final decision:
  100 = every citation verified, cause clear, no open questions
    0 = citations unsupported or cause entirely unresolved
Respond with ONLY JSON:
{
  "all_supported": true,
  "unsupported_points": [],
  "final_decision": "COVERED | PARTIAL | DENIED | ABSTAIN",
  "confidence": <integer 0-100>,
  "notes": "one or two sentences"
}"""


# --------------------------------------------------------------------------- #
# LLM client wrapper. dry_run returns canned JSON so the whole pipeline runs
# with no key; real mode calls the Anthropic Messages API.
# --------------------------------------------------------------------------- #
_MOCKS = {
    "plan": json.dumps({
        "queries": [
            "sudden accidental water discharge coverage",
            "flood surface water exclusion endorsement",
            "water backup sewer drain sublimit",
            "basement water heavy rain prior claim",
        ],
        "tools": ["weather_cat", "sanctions_screen"],
        "reasoning": "Water loss after heavy rain needs coverage, exclusions, backup add-on, and weather data.",
    }),
    "rerank": json.dumps({
        "ranked_ids": ["END-HO217", "END-HO305", "POL-BASE", "GUIDE-WTR", "CLM-2201", "DEF-FLOOD"]
    }),
    "draft": json.dumps({
        "decision": "PARTIAL",
        "confidence": 42,
        "rationale": (
            "The base policy covers sudden and accidental water discharge (POL-BASE), "
            "but endorsement HO-217 excludes loss from flood or surface water (END-HO217), "
            "and the weather feed shows a flood event on the loss date. If the intrusion "
            "instead came from a sewer/drain backup, HO-305 provides up to a $5,000 sublimit "
            "(END-HO305). Because the proximate cause is unconfirmed, coverage is partial "
            "pending cause determination, consistent with the water-loss guideline (GUIDE-WTR)."
        ),
        "citations": [
            {"doc_id": "POL-BASE", "point": "Covers sudden and accidental water discharge"},
            {"doc_id": "END-HO217", "point": "Excludes flood and surface water"},
            {"doc_id": "END-HO305", "point": "Adds water-backup coverage up to $5,000"},
            {"doc_id": "GUIDE-WTR", "point": "Confirm proximate cause and pull weather data"},
        ],
        "open_questions": [
            "Was the intrusion surface water (excluded) or a drain/sewer backup (covered to $5,000)?",
            "Is there evidence of the entry path (foundation crack vs. floor drain)?",
        ],
    }),
    "verify": json.dumps({
        "all_supported": True,
        "unsupported_points": [],
        "final_decision": "PARTIAL",
        "confidence": 42,
        "notes": "All cited points trace to the referenced documents; cause of loss remains an open question for the adjuster.",
    }),
}


@dataclass
class LLM:
    dry_run: bool = False
    client: object = field(default=None, repr=False)

    def __post_init__(self):
        if not self.dry_run:
            import anthropic  # imported lazily so dry-run needs nothing
            self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def complete(self, *, step: str, system: str, user: str, model: str) -> str:
        if self.dry_run:
            return _MOCKS[step]
        resp = self.client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def parse_json(s: str) -> dict:
    s = re.sub(r"```(json)?|```", "", s).strip()
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1:
        s = s[a:b + 1]
    return json.loads(s)


# --------------------------------------------------------------------------- #
# Confidence visualiser — generates a PNG dashboard of the pipeline scores.
# --------------------------------------------------------------------------- #
DECISION_COLORS = {"COVERED": "#2ecc71", "PARTIAL": "#f39c12", "DENIED": "#e74c3c", "ABSTAIN": "#95a5a6"}


def _gauge(ax, score: int, title: str, decision: str) -> None:
    """Draw a half-circle speedometer gauge on *ax*."""
    color = DECISION_COLORS.get(decision, "#3498db")
    low_color = "#e74c3c"
    mid_color = "#f39c12"
    high_color = "#2ecc71"

    # Background arc segments (LOW / MED / HIGH zones)
    for (start, end, fc) in [(0, 40, low_color), (40, 70, mid_color), (70, 100, high_color)]:
        theta1 = 180 - start * 1.8
        theta2 = 180 - end * 1.8
        wedge = mpatches.Wedge((0.5, 0.3), 0.38, theta2, theta1,
                               width=0.14, facecolor=fc, alpha=0.25, transform=ax.transAxes)
        ax.add_patch(wedge)

    # Filled arc showing actual score
    filled_wedge = mpatches.Wedge((0.5, 0.3), 0.38, 180 - score * 1.8, 180,
                                  width=0.14, facecolor=color, alpha=0.85, transform=ax.transAxes)
    ax.add_patch(filled_wedge)

    # Needle
    import math
    angle_deg = 180 - score * 1.8
    angle_rad = math.radians(angle_deg)
    nx = 0.5 + 0.28 * math.cos(angle_rad)
    ny = 0.3 + 0.28 * math.sin(angle_rad)
    ax.annotate("", xy=(nx, ny), xytext=(0.5, 0.3),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2))

    # Centre dot
    circle = plt.Circle((0.5, 0.3), 0.025, color="#2c3e50", transform=ax.transAxes, zorder=5)
    ax.add_patch(circle)

    # Score text
    ax.text(0.5, 0.58, f"{score}/100", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color, transform=ax.transAxes)
    label = "HIGH" if score >= 70 else ("MED" if score >= 40 else "LOW")
    ax.text(0.5, 0.70, label, ha="center", va="center",
            fontsize=13, color=color, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.82, title, ha="center", va="center",
            fontsize=10, color="#555", transform=ax.transAxes)
    ax.text(0.5, 0.10, decision, ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="white", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"))

    # Zone labels
    ax.text(0.09, 0.30, "LOW", ha="center", fontsize=7, color=low_color, transform=ax.transAxes)
    ax.text(0.50, 0.72, "", ha="center", fontsize=7, transform=ax.transAxes)  # spacer
    ax.text(0.91, 0.30, "HIGH", ha="center", fontsize=7, color=high_color, transform=ax.transAxes)
    ax.axis("off")


def _horizontal_bar(ax, score: int, title: str, decision: str) -> None:
    """Draw a segmented horizontal bar showing score vs 100."""
    color = DECISION_COLORS.get(decision, "#3498db")
    ax.barh(0, 100, color="#ecf0f1", height=0.5)
    ax.barh(0, score, color=color, height=0.5, alpha=0.85)
    # Zone dividers
    for x, lbl in [(40, "40"), (70, "70")]:
        ax.axvline(x, color="#bdc3c7", lw=1, ls="--")
        ax.text(x, 0.42, lbl, ha="center", va="bottom", fontsize=8, color="#7f8c8d")
    ax.text(score, 0, f"  {score}", va="center", fontsize=11, fontweight="bold", color=color)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.5, 0.8)
    ax.set_yticks([])
    ax.set_xlabel("Confidence (0 = low  →  100 = high)", fontsize=9)
    ax.set_title(f"{title}  [{decision}]", fontsize=10, fontweight="bold",
                 color=DECISION_COLORS.get(decision, "#2c3e50"))
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_confidence_chart(
    draft_score: int,
    draft_decision: str,
    verify_score: int,
    verify_decision: str,
    output_path: str = "confidence_report.png",
) -> str:
    """Build a 2-panel confidence dashboard and save to *output_path*."""
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor("#f8f9fa")

    # Title banner
    fig.text(0.5, 0.96, "Claims Coverage Copilot — Confidence Dashboard",
             ha="center", va="top", fontsize=15, fontweight="bold", color="#2c3e50")
    fig.text(0.5, 0.91, "Scale: 0 = low confidence (unresolved cause / weak evidence)  "
             "→  100 = high confidence (clear policy + confirmed cause)",
             ha="center", va="top", fontsize=9, color="#7f8c8d")

    # Top row: two gauges
    ax1 = fig.add_axes([0.05, 0.38, 0.40, 0.48])
    ax2 = fig.add_axes([0.55, 0.38, 0.40, 0.48])
    _gauge(ax1, draft_score, "Draft Determination", draft_decision)
    _gauge(ax2, verify_score, "Verified Decision", verify_decision)

    # Bottom row: two horizontal bars
    ax3 = fig.add_axes([0.08, 0.15, 0.37, 0.16])
    ax4 = fig.add_axes([0.58, 0.15, 0.37, 0.16])
    _horizontal_bar(ax3, draft_score, "Draft", draft_decision)
    _horizontal_bar(ax4, verify_score, "Verified", verify_decision)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=d, alpha=0.85)
                      for d, c in DECISION_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# --------------------------------------------------------------------------- #
# The agentic pipeline.
# --------------------------------------------------------------------------- #
def run(claim: str, retriever: HybridRetriever, llm: LLM) -> dict:
    line = lambda c="-": print(c * 78)

    # 1. PLAN
    plan = parse_json(llm.complete(
        step="plan", system=PLANNER_SYSTEM, model=REASONING_MODEL,
        user=f"<claim>\n{claim}\n</claim>",
    ))
    print("1) PLAN"); line()
    print("   queries:", plan["queries"])
    print("   tools  :", plan.get("tools", []))

    # 2. RETRIEVE (hybrid BM25 + TF-IDF + RRF)
    candidates = retriever.search(plan["queries"])
    print("\n2) RETRIEVE (hybrid + RRF)"); line()
    for d in candidates:
        print(f"   [{d['id']:<9}] {d['source']}")

    # 3. RERANK (LLM)
    cand_block = "\n".join(f"[{d['id']}] {d['text']}" for d in candidates)
    rr = parse_json(llm.complete(
        step="rerank", system=RERANK_SYSTEM, model=RERANK_MODEL,
        user=f"<claim>\n{claim}\n</claim>\n<candidates>\n{cand_block}\n</candidates>",
    ))
    cand_ids = {d["id"] for d in candidates}
    ordered = [i for i in rr.get("ranked_ids", []) if i in cand_ids]
    ordered += [i for i in cand_ids if i not in ordered]      # robustness
    evidence = [ID_TO_DOC[i] for i in ordered[:TOP_N_EVIDENCE]]
    print("\n3) RERANK -> evidence kept:", [d["id"] for d in evidence])

    # 4. REAL-TIME TOOLS
    realtime = {}
    print("\n4) REAL-TIME TOOLS"); line()
    for name in plan.get("tools", []):
        if name == "weather_cat":
            realtime[name] = TOOLS[name]("2026-03-14", "Austin, TX")
        elif name == "sanctions_screen":
            realtime[name] = TOOLS[name]("John Roe")
        print(f"   {name}: {realtime[name]}")

    # 5. DRAFT determination (LLM, grounded + cited)
    ev_block = "\n".join(f"[{d['id']}] ({d['source']}) {d['text']}" for d in evidence)
    draft = parse_json(llm.complete(
        step="draft", system=DRAFTER_SYSTEM, model=REASONING_MODEL,
        user=(f"<claim>\n{claim}\n</claim>\n<evidence>\n{ev_block}\n</evidence>\n"
              f"<realtime>\n{json.dumps(realtime)}\n</realtime>"),
    ))

    # 6. VERIFY (faithfulness gate -> abstain if unsupported)
    verify = parse_json(llm.complete(
        step="verify", system=VERIFIER_SYSTEM, model=REASONING_MODEL,
        user=(f"<draft>\n{json.dumps(draft)}\n</draft>\n<evidence>\n{ev_block}\n</evidence>"),
    ))

    def _conf_bar(score: int, width: int = 20) -> str:
        filled = round(score / 100 * width)
        bar = "#" * filled + "-" * (width - filled)
        label = "HIGH" if score >= 70 else ("MED" if score >= 40 else "LOW")
        return f"[{bar}] {score}/100 ({label})"

    # ---- report ----
    print("\n5) DRAFT DETERMINATION"); line()
    print("   decision   :", draft["decision"])
    print("   confidence :", _conf_bar(draft.get("confidence", 0)))
    print("   rationale  :", draft["rationale"])
    print("   citations:")
    for c in draft["citations"]:
        print(f"       - [{c['doc_id']}] {c['point']}")
    print("   open questions:")
    for q in draft["open_questions"]:
        print("       -", q)

    print("\n6) VERIFIER (faithfulness gate)"); line()
    print("   all_supported :", verify["all_supported"])
    print("   FINAL DECISION:", verify["final_decision"])
    print("   confidence    :", _conf_bar(verify.get("confidence", 0)))
    print("   notes         :", verify["notes"])
    line("=")
    print("Human adjuster reviews this draft before any decision is communicated.")

    chart_path = render_confidence_chart(
        draft_score=draft.get("confidence", 0),
        draft_decision=draft["decision"],
        verify_score=verify.get("confidence", 0),
        verify_decision=verify["final_decision"],
    )
    print(f"\nConfidence chart saved -> {chart_path}")

    return {"plan": plan, "evidence": [d["id"] for d in evidence],
            "realtime": realtime, "draft": draft, "verify": verify}


def main() -> None:
    global REASONING_MODEL
    ap = argparse.ArgumentParser(description="Advanced-RAG insurance claims copilot")
    ap.add_argument("--dry-run", action="store_true",
                    help="run offline with canned LLM responses (no API key needed)")
    ap.add_argument("--claim", default=DEFAULT_CLAIM, help="FNOL / claim text")
    ap.add_argument("--model", default=REASONING_MODEL, help="reasoning model id")
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Either run with --dry-run, or set the key:\n"
            '    export ANTHROPIC_API_KEY="sk-ant-..."   (macOS/Linux)\n'
            '    setx   ANTHROPIC_API_KEY "sk-ant-..."    (Windows, new shell)'
        )

    REASONING_MODEL = args.model
    mode = "DRY-RUN (offline, canned LLM)" if args.dry_run else f"LIVE API ({REASONING_MODEL})"
    print("=" * 78)
    print(f"CLAIMS COVERAGE COPILOT  —  mode: {mode}")
    print("=" * 78)
    print("CLAIM:\n  " + args.claim.replace("\n", "\n  ") + "\n")

    retriever = HybridRetriever(CORPUS)
    llm = LLM(dry_run=args.dry_run)
    run(args.claim, retriever, llm)


if __name__ == "__main__":
    main()
