# ============================
# Imports
# ============================
import os
import re
import json
from typing import List, Dict

import streamlit as st
import wikipedia
from openai import OpenAI

# ============================
# Step 1: Input validation & normalisation
# ============================

def ensure_industry_context(text: str) -> str:
    """
    If the user does not explicitly mention industry/market/sector,
    append 'industry' to make the query industry-level.
    """
    t = text.lower()
    if any(k in t for k in ["industry", "market", "sector", "value chain"]):
        return text.strip()
    return f"{text.strip()} industry"


def llm_validate_and_fix_industry(user_text: str) -> Dict[str, str]:
    """
    Uses an LLM to:
    - decide whether the input represents an industry
    - correct typos or incomplete phrasing
    - reject brand/company-level inputs
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "is_industry": "true",
            "corrected": user_text.strip(),
            "message": "LLM validation skipped (no API key).",
            "suggestions": [],
        }

    client = OpenAI(api_key=api_key)

    system = (
        "You validate user inputs for an industry market research tool. "
        "Decide whether the input is an industry/market/sector concept. "
        "Correct typos and incomplete phrasing. "
        "Reject brands, companies, or products."
    )

    user = f"""
User input: "{user_text}"

Return ONLY valid JSON with:
- is_industry: true/false
- corrected: corrected industry query (max 6 words)
- message: short explanation
- suggestions: 2-3 alternative industry queries (if not industry)
""".strip()

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    raw = resp.choices[0].message.content or ""
    
    # Extract first JSON object from the response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    
    if not match:
        return {
            "is_industry": "false",
            "corrected": "",
            "message": "Could not validate input. Please re-enter a clearer industry.",
            "suggestions": [],
        }
    
    try:
        data = json.loads(match.group())
    except Exception:
        return {
            "is_industry": "false",
            "corrected": "",
            "message": "Could not validate input. Please re-enter a clearer industry.",
            "suggestions": [],
        }
    
    # Normalise fields safely
    return {
        "is_industry": str(data.get("is_industry", "false")).lower(),
        "corrected": str(data.get("corrected", "")).strip(),
        "message": str(data.get("message", "")).strip(),
        "suggestions": data.get("suggestions", []) if isinstance(data.get("suggestions", []), list) else [],
    }



# ============================
# Step 2: Wikipedia retrieval (use Wikipedia ranking)
# ============================

def _is_bad_title(title: str) -> bool:
    """Filter out disambiguation pages."""
    t = title.lower()
    return "(disambiguation)" in t or t == "disambiguation"

def title_matches_query(title: str, query: str) -> bool:
    """
    Title-only filter: keep titles that contain at least one key query token.
    """
    t = title.lower()
    tokens = [x for x in re.findall(r"[a-z0-9]+", query.lower()) if len(x) >= 4]

    # Avoid over-filtering very short queries
    if not tokens:
        return True

    return any(tok in t for tok in tokens)

def search_wikipedia(industry: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Retrieve up to `limit` Wikipedia pages using Wikipedia's own ranking.
    """
    wikipedia.set_lang("en")

    queries = [
        industry,
        f"{industry} market",
        f"{industry} sector",
        f"global {industry}",
        f"{industry} supply chain",
        f"{industry} value chain",
    ]

    seen = set()
    pages: List[Dict[str, str]] = []

    for q in queries:
        try:
            results = wikipedia.search(q, results=10)
        except Exception:
            continue

        for title in results:
            if title in seen or _is_bad_title(title):
                continue
            if not title_matches_query(title, industry):
                continue
            try:
                page = wikipedia.page(title, auto_suggest=False, redirect=True)
                if len(page.content or "") < 800:
                    continue

                pages.append({
                    "title": page.title,
                    "url": page.url,
                    "content": page.content[:3000],  # cost control
                })
                seen.add(title)
            except Exception:
                continue

            if len(pages) >= limit:
                return pages

    return pages


# ============================
# Step 3: Reporting helpers
# ============================

def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def trim_to_word_limit(text: str, limit: int) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit])


def word_count(text: str) -> int:
    return len(text.split())


def embed_reference_links(text: str, ref_map: Dict[str, str]) -> str:
    """Convert [1][2] → [1], [2] and embed clickable links."""
    text = re.sub(r"\]\[", "], [", text)
    for k, url in ref_map.items():
        text = re.sub(rf"\[{k}\]", f"[{k}]({url})", text)
    return text


def build_llm_report(industry: str, pages: List[Dict[str, str]]) -> str:
    """Generate a structured industry report using GPT-4o."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    refs = {str(i): p["url"] for i, p in enumerate(pages, start=1)}
    snippets = []

    for i, p in enumerate(pages, start=1):
        snippet = " ".join(split_sentences(p["content"])[:4])
        snippets.append(f"[{i}] {p['title']}\nURL: {p['url']}\n{snippet}")

    prompt = f"""
Write an industry report for: {industry}

Rules:
- Under 450 words
- Use ONLY the snippets below
- Use **bold headers** exactly as shown
- Cite sources as [1], [2], etc.

**Overview**
**Market structure / value chain**
**Key segments / products**
**Demand drivers & constraints**
**Trends / near-term outlook**
**Key takeaways**

SNIPPETS:
{chr(10).join(snippets)}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )

    text = resp.choices[0].message.content.strip()
    text = embed_reference_links(text, refs)
    return trim_to_word_limit(text, 500)


# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write("Provide an industry to generate a Wikipedia-based market research report.")

industry = st.text_input("Industry")

if st.button("Generate report"):
    if not industry.strip():
        st.warning("Please enter an industry.")
        st.stop()

    # LLM validation & correction
    result = llm_validate_and_fix_industry(industry)
    if result["is_industry"] != "true":
        st.warning(result["message"])
        for s in result.get("suggestions", []):
            st.write(f"- {s}")
        st.stop()

    industry = ensure_industry_context(result["corrected"])

    st.subheader("Step 1: Industry validated ✅")
    st.write(f"Interpreted query: **{industry}**")

    pages = search_wikipedia(industry)

    if len(pages) < 3:
        st.error("Not enough relevant Wikipedia pages found. Please try a clearer industry.")
        st.stop()
    if len(pages) < 5:
        st.warning(f"Only {len(pages)} relevant pages found. Report uses available sources.")

    st.subheader("Step 2: Relevant Wikipedia Pages")
    for p in pages:
        st.write(p["url"])

    st.subheader("Step 3: Industry Report")
    report = build_llm_report(industry, pages)
    st.markdown(report)
    st.caption(f"Word count: {word_count(report)} / 500")
