import os
import re
from typing import List, Dict

import streamlit as st
import wikipedia
import json
from openai import OpenAI
from difflib import get_close_matches

# ----------------------------
# Step 1 helpers: validation & normalisation
# ----------------------------

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

def ensure_industry_context(text: str) -> str:
    """
    If user does not specify industry/market/sector,
    automatically append 'industry'.
    """
    t = text.strip().lower()
    industry_terms = ["industry", "market", "sector", "value chain"]

    if any(term in t for term in industry_terms):
        return text.strip()

    return f"{text.strip()} industry"

def llm_validate_and_fix_industry(user_text: str) -> Dict[str, str]:
    """
    Uses an LLM to:
    - decide if input is a valid industry/market/sector concept
    - correct typos / incomplete inputs into a better industry query
    If not valid, returns suggestions and asks user to re-input.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # No API key: fall back to original text (no LLM validation)
        return {
            "is_industry": "true",
            "corrected": user_text.strip(),
            "message": "LLM validation skipped (no OPENAI_API_KEY)."
        }

    client = OpenAI(api_key=api_key)

    system = (
        "You validate user inputs for an industry market research tool. "
        "Your job: decide if the user input represents an industry/market/sector. "
        "If the input is a company/brand/product/person/place or too vague, mark it NOT an industry. "
        "If it's close, correct typos and rewrite into a clear industry query."
    )

    user = f"""
User input: "{user_text}"

Return ONLY valid JSON (no markdown) with keys:
- is_industry: true/false
- corrected: string (if is_industry true, a corrected industry query, e.g. "bubble tea market", "luxury sports car industry")
- message: short string explaining what you did
- suggestions: array of 2-3 alternative industry queries (strings)

Rules:
- If it looks like a brand/company (e.g., "Lamborghini"), set is_industry=false and suggest the market/industry instead.
- Correct obvious typos (e.g., "insurence" -> "insurance").
- If too generic (e.g., "food"), make it more industry-like (e.g., "food industry" or "food and beverage industry").
- Keep it short (max 6 words for corrected).
""".strip()

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # GPT-5 mini: no temperature; use max_completion_tokens
        max_completion_tokens=400,
    )

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        return {
            "is_industry": "false",
            "corrected": "",
            "message": "Validation returned empty output. Please re-enter a clearer industry.",
            "suggestions": []
        }

    try:
        data = json.loads(raw)
    except Exception:
        # If model returns slightly non-JSON, fail safely
        return {
            "is_industry": "false",
            "corrected": "",
            "message": "Could not parse validation result. Please re-enter a clearer industry.",
            "suggestions": []
        }

    # Normalise types
    data["is_industry"] = str(data.get("is_industry", "false")).lower()
    data["corrected"] = str(data.get("corrected", "")).strip()
    data["message"] = str(data.get("message", "")).strip()
    data["suggestions"] = data.get("suggestions", [])
    if not isinstance(data["suggestions"], list):
        data["suggestions"] = []

    return data


# ----------------------------
# Step 2 helpers: Wikipedia retrieval
# ----------------------------

def _is_bad_title(title: str) -> bool:
    t = title.lower().strip()
    return "(disambiguation)" in t or t == "disambiguation"

def search_wikipedia(industry: str, limit: int = 5) -> List[Dict[str, str]]:
    wikipedia.set_lang("en")

    queries = [
        industry,
        f"{industry} industry",
        f"{industry} market",
        f"global {industry}",
        f"{industry} value chain",
    ]

    seen_titles = set()
    results: List[Dict[str, str]] = []

    for q in queries:
        try:
            search_results = wikipedia.search(q, results=10)
        except Exception:
            continue

        for title in search_results:
            if title in seen_titles or _is_bad_title(title):
                continue

            try:
                page = wikipedia.page(title, auto_suggest=False, redirect=True)
                content = (page.content or "").strip()
                if len(content) < 800:
                    continue

                results.append({
                    "title": page.title,
                    "url": page.url,
                    "content": content[:3000],  # cost control
                })
                seen_titles.add(title)
            except Exception:
                continue

            if len(results) >= limit:
                return results  # IMPORTANT: keep Wikipedia’s ranking order

    return results

# ----------------------------
# Step 3 helpers: reporting
# ----------------------------
def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]

def trim_to_word_limit(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def word_count(text: str) -> int:
    return len(text.split())

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
    
    # Build reference mapping for citations
    reference_map = {
        str(i): page["url"]
        for i, page in enumerate(pages, start=1)
    }

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
- Use citation markers like [1], [2], etc.
- Each citation MUST be written exactly as: [1], [2], etc. (no text inside).
- Citations will be converted into clickable links automatically.
- Use professional business language.

Formatting rules (IMPORTANT):
- Each section header MUST be on its own line.
- Each section header MUST be in **bold Markdown**.
- Insert a blank line after each header.
- Do NOT include numbering or parentheses in headers.
- Content must start on the line AFTER the header.

Required structure (use EXACT headers):

**Overview**

**Market structure / value chain**

**Key segments / products**

**Demand drivers & constraints**

**Trends / near-term outlook**

**Key takeaways**
(use bullet points)

WIKIPEDIA SNIPPETS:
{sources}
""".strip()


    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=2000,
    )


    text = resp.choices[0].message.content.strip()
    
    # Embed clickable reference links
    text = embed_reference_links(text, reference_map)
    
    return trim_to_word_limit(text, 500)


def embed_reference_links(text: str, ref_map: Dict[str, str]) -> str:
    """
    Converts [1][2] into [1], [2] with clickable Markdown links.
    """
    # First, insert a comma + space between adjacent citations
    text = re.sub(r"\]\[", "], [", text)

    # Then convert each [n] into a markdown link
    for ref_id, url in ref_map.items():
        text = re.sub(
            rf"\[{re.escape(ref_id)}\]",
            f"[{ref_id}]({url})",
            text
        )
    return text



# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Market Research Assistant", layout="centered")

st.title("Market Research Assistant")
st.write("Provide an industry to generate a Wikipedia-based market research report.")

industry = st.text_input("Industry")

if st.button("Generate report"):
    # Step 1: basic validation
    if not is_valid_industry(industry):
        st.warning("Please enter a valid industry (e.g. 'insurance', 'bubble tea', 'electric vehicles').")
        st.stop()

    # Step 1b: LLM validate + correct (typos/incomplete/not industry)
    v = llm_validate_and_fix_industry(industry)

    if v["is_industry"] != "true":
        st.warning(v.get("message") or "This does not look like an industry. Please re-enter.")
        sugg = v.get("suggestions", [])
        if sugg:
            st.write("Try one of these instead:")
            for s in sugg:
                st.write(f"- {s}")
        st.stop()

    # Use corrected query
    industry = v["corrected"] if v["corrected"] else industry.strip()

    # Step 1c: ensure industry context (append industry if missing)
    industry = ensure_industry_context(industry)

    st.subheader("Step 1: Industry validated ✅")
    st.write(f"Interpreted industry query: **{industry}**")


    pages = search_wikipedia(industry.strip(), limit=5)

    if len(pages) < 5:
        st.warning(
            f"Only {len(pages)} relevant Wikipedia pages were found for this query. "
            "The report will be generated using the available sources."
        )
        partial = True
    else:
        partial = False
    
    if len(pages) < 3:
        st.error(
            "Not enough relevant Wikipedia pages were found to generate a reliable report. "
            "Try a different spelling or a more specific industry term (e.g., 'bubble tea', 'bubble tea market')."
        )
        st.stop()


    if not pages:
        st.info("No Wikipedia pages found for that industry. Try a different industry term.")
        st.stop()

    st.subheader("Step 2: Top 5 Relevant Wikipedia Pages")
    if partial:
        st.info(
            "Fewer than five highly relevant industry-level Wikipedia pages were available. "
            "The report below is generated using the most relevant sources found."
        )

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
    st.markdown(report)
    st.caption(f"Word count: {word_count(report)} (max 500)")
