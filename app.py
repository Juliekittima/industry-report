import os
import re
from typing import List, Dict

import streamlit as st
import wikipedia
from openai import OpenAI

BAD_INPUTS = {"", "hi", "hello", "test", "asdf", "help", "idk", "none"}

def is_valid_industry(text: str) -> bool:
    t = (text or "").strip().lower()
    if t in BAD_INPUTS:
        return False
    if len(t) < 3:
        return False
    if re.fullmatch(r"[\d\W_]+", t):
        return False
    return True

def _is_bad_title(title: str) -> bool:
    t = title.lower().strip()
    return "(disambiguation)" in t or t == "disambiguation"

def search_wikipedia(industry: str, limit: int = 5) -> List[Dict[str, str]]:
    wikipedia.set_lang("en")
    queries = [industry, f"{industry} industry", f"{industry} market"]

    seen_titles = set()
    candidates: List[Dict[str, str]] = []

    for q in queries:
        try:
            results = wikipedia.search(q, results=10)
        except Exception:
            continue

        for title in results:
            if title in seen_titles or _is_bad_title(title):
                continue
            try:
                page = wikipedia.page(title, auto_suggest=False, redirect=True)
                content = (page.content or "").strip()
                if len(content) < 800:
                    continue
                candidates.append({"title": page.title, "url": page.url, "content": content[:4000]})
                seen_titles.add(title)
            except Exception:
                continue

            if len(candidates) >= 15:
                break
        if len(candidates) >= 15:
            break

    # rank by longer content (simple + explainable)
    candidates.sort(key=lambda x: len(x["content"]), reverse=True)
    return candidates[:limit]


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def build_extractive_report(industry: str, pages: List[Dict[str, str]]) -> str:
    lines = []
    lines.append(f"Industry report: {industry}")
    lines.append("")

    if pages:
        lines.append("Overview")
        for page in pages:
            title = page["title"]
            extract = page.get("content", "")
            sentences = split_sentences(extract)[:2]
            if sentences:
                summary = " ".join(sentences)
                lines.append(f"- {title}: {summary}")
            else:
                lines.append(f"- {title}: No summary available from Wikipedia.")
        lines.append("")

    lines.append("Key takeaways")
    lines.append("- The pages above describe the industry’s scope, structure, and related concepts.")
    lines.append("- Use the linked Wikipedia sources to validate definitions, terminology, and historical context.")
    lines.append("- Consider augmenting this baseline with market size, growth rates, and competitive data from proprietary sources.")

    report = "\n".join(lines).strip()
    return trim_to_word_limit(report, 500)


def build_llm_report(industry: str, pages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    client = OpenAI(api_key=api_key)

    # Build short grounded snippets (cost control)
    doc_blurbs = []
    for i, page in enumerate(pages, start=1):
        title = page["title"]
        url = page["url"]
        content = page.get("content", "")
        snippet = " ".join(split_sentences(content)[:4])
        doc_blurbs.append(f"[{i}] {title}\nURL: {url}\nSNIPPET: {snippet}\n")

    sources = "\n".join(doc_blurbs)

    system = (
        "You are a market research assistant. "
        "You must ONLY use the provided Wikipedia snippets. "
        "If something is not in the snippets, say it is not available in the sources. "
        "Do not invent numbers, market sizes, or company names. "
        "Keep it concise and business-like."
    )

    user = f"""
Write an industry report for: {industry}

Requirements:
- Under 450 words (hard limit: must be under 500).
- Use ONLY the snippets below.
- Include citations [1]..[5] to show which snippet supports each point.
- Structure:
  1) Overview (1-2 sentences)
  2) Market structure / value chain
  3) Key segments / products
  4) Demand drivers & constraints
  5) Trends / near-term outlook
  6) 3-5 bullet takeaways

WIKIPEDIA SNIPPETS:
{sources}
""".strip()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_completion_tokens=650,
    )


    return trim_to_word_limit(resp.choices[0].message.content.strip(), 500)



def trim_to_word_limit(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def word_count(text: str) -> int:
    return len(text.split())


st.set_page_config(page_title="Market Research Assistant", layout="centered")

st.title("Market Research Assistant")
st.write("Provide an industry to generate a Wikipedia-based market research report.")

industry = st.text_input("Industry")

if st.button("Generate report"):
    # Step 1: Industry validation (Q1)
    if not is_valid_industry(industry):
        st.warning(
            "Please enter a valid industry (e.g. 'electric vehicles', 'insurance', 'UK coffee shops')."
        )
        st.stop()

    try:
        pages = search_wikipedia(industry.strip(), limit=5)
    except Exception as exc:
        st.error(f"Wikipedia search failed: {exc}")
        st.stop()

    if not pages:
        st.info("No Wikipedia pages found for that industry. Try a different industry term.")
        st.stop()

    st.subheader("Step 2: Top 5 Relevant Wikipedia Pages")
    for page in pages:
        st.write(page["url"])

    use_llm = bool(os.getenv("OPENAI_API_KEY"))
    if use_llm:
        try:
            report = build_llm_report(industry.strip(), pages)
            st.caption("Report generated with LLM summarization.")
        except Exception as exc:
            st.warning(f"LLM summarization failed, falling back to extractive summary: {exc}")
            report = build_extractive_report(industry.strip(), pages)
    else:
        report = build_extractive_report(industry.strip(), pages)
        st.caption("Set OPENAI_API_KEY to enable LLM summarization.")

    st.subheader("Step 3: Industry Report")
    st.write(report)
    st.caption(f"Word count: {word_count(report)} (max 500)")
