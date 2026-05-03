import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


def generate_citation(document: dict) -> str:
    """Generate a simple LexisNexis-style citation string."""
    case_title = document.get("case_title") or ""
    reporters = document.get("reporters") or []
    court = document.get("case_court") or ""
    case_date = document.get("case_date") or ""

    year = ""
    if case_date:
        match = re.search(r"\b(18|19|20)\d{2}\b", case_date)
        if match:
            year = match.group()

    reporters_str = ", ".join(reporters)

    citation_parts = []

    if case_title:
        citation_parts.append(case_title)

    if reporters_str:
        citation_parts.append(reporters_str)

    citation = ", ".join(citation_parts)

    if court and year:
        citation += f" ({court}. {year})"
    elif year:
        citation += f" ({year})"

    if citation:
        citation += ". LexisNexis."
    else:
        citation = "LexisNexis."

    return citation


def extract_single_paragraph(soup: BeautifulSoup, start_id: str) -> str | None:
    """Extract the first paragraph after a given section anchor."""
    anchor = soup.find("span", id=start_id)
    if not anchor:
        return None

    p = anchor.find_next("p")
    if p:
        return p.get_text(" ", strip=True)

    return None


def parse_lexis_html(html: str) -> dict:
    """Parse LexisNexis HTML and return extracted fields as a dictionary."""
    soup = BeautifulSoup(html, "lxml")

    # =========================
    # case_no
    # =========================
    case_no = None

    for p in soup.find_all("p", class_="SS_DocumentInfo"):
        text = p.get_text(strip=True)
        if re.search(r"No\.\s*\d+", text):
            case_no = text
            break

    # =========================
    # case_title
    # =========================
    title_tag = soup.find("h1", id="SS_DocumentTitle")
    case_title = title_tag.get_text(strip=True) if title_tag else None

    # =========================
    # case_date
    # =========================
    case_date = None

    for p in soup.find_all("p", class_="SS_DocumentInfo"):
        text = p.get_text(strip=True)
        if "Argued" in text or "Decided" in text:
            case_date = text
            break

    # =========================
    # court
    # =========================
    court_name = None

    for p in soup.find_all("p", class_="SS_DocumentInfo"):
        text = p.get_text(strip=True)
        if "Court" in text:
            court_name = text
            break

    # =========================
    # reporters
    # =========================
    reporters = []

    rptr_container = soup.find("span", class_="SS_RptrLine")

    if rptr_container:
        for tag in rptr_container.find_all(["a", "span"]):
            if tag.name == "span" and "SS_NonPaginatedRptr" not in tag.get("class", []):
                continue

            text = tag.get_text(" ", strip=True)
            text = re.sub(r"\*+", "", text)
            text = re.sub(r"\[\*+\d+\]", "", text).strip()

            if text and text not in reporters:
                reporters.append(text)

    # =========================
    # prior_history
    # =========================
    prior_history = None

    ph_anchor = soup.find("span", id="JUMPTO_PriorHistory")

    if ph_anchor:
        texts = []

        for sibling in ph_anchor.find_all_next():
            if sibling.name == "span" and sibling.get("id", "").startswith("JUMPTO_"):
                break

            if sibling.name == "p":
                txt = sibling.get_text(" ", strip=True)
                if txt:
                    texts.append(txt)

        prior_history = "\n".join(texts) if texts else None

    # =========================
    # core_terms
    # =========================
    core_terms_list = []

    core_anchor = soup.find("h2", id="JUMPTO_CoreTerms")

    if core_anchor:
        for sibling in core_anchor.next_siblings:
            if isinstance(sibling, str):
                text = sibling.strip()

                if text and text != "Core Terms":
                    core_terms_list = [t.strip() for t in text.split(",") if t.strip()]
                    break

    # =========================
    # Case Summary
    # =========================
    case_summary = {
        "procedural_posture": extract_single_paragraph(soup, "JUMPTO_ProceduralPosture"),
        "overview": extract_single_paragraph(soup, "JUMPTO_Overview"),
        "outcome": extract_single_paragraph(soup, "JUMPTO_Outcome"),
    }

    # =========================
    # Counsel
    # =========================
    counsel_list = []

    counsel_anchor = soup.find("span", id="JUMPTO_Counsel")

    if counsel_anchor:
        current = []

        for node in counsel_anchor.next_siblings:
            if (
                getattr(node, "name", None) == "span"
                and node.get("id", "").startswith("JUMPTO_")
            ):
                break

            if getattr(node, "name", None) == "br":
                if current:
                    text = " ".join(current).strip()
                    if text:
                        counsel_list.append(text)
                    current = []
                continue

            if isinstance(node, str):
                cleaned = node.replace("\xa0", " ").strip()
                if cleaned:
                    current.append(cleaned)

        if current:
            counsel_list.append(" ".join(current).strip())

    # =========================
    # Opinion by
    # =========================
    opinion_by = None

    opinion_anchor = soup.find("span", id="JUMPTO_Opinionby")

    if opinion_anchor:
        judge_tag = opinion_anchor.find_next("a", class_="SS_EntityLink")

        if judge_tag:
            opinion_by = {
                "name": judge_tag.get_text(strip=True),
                "link": judge_tag.get("data-judgepath"),
            }

    # =========================
    # Opinion
    # =========================
    opinion_paragraphs = []

    opinion_anchor = soup.find("h2", id="JUMPTO_Opinion")

    if opinion_anchor:
        for node in opinion_anchor.find_all_next():
            if (
                getattr(node, "name", None) in ["h2", "span"]
                and node.get("id", "").startswith("JUMPTO_")
                and node != opinion_anchor
            ):
                break

            if node.name == "p":
                text = node.get_text(" ", strip=True)

                if text:
                    opinion_paragraphs.append(text)

    opinion_text = "\n\n".join(opinion_paragraphs)

    # =========================
    # Final document
    # =========================
    document = {
        "case_no": case_no,
        "case_title": case_title,
        "case_date": case_date,
        "case_court": court_name,
        "reporters": reporters,
        "prior_history": prior_history,
        "core_terms": core_terms_list,
        "case_summary": case_summary,
        "counsel": counsel_list,
        "opinion_by": opinion_by,
        "opinion": opinion_text,
    }

    # Add citation into output
    document["citation"] = generate_citation(document)

    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a LexisNexis HTML file and export extracted case metadata."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input LexisNexis HTML file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="lexis_document_parsed.json",
        help="Path to the output JSON file. Default: lexis_document_parsed.json",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        html = f.read()

    document = parse_lexis_html(html)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    print("HTML 讀取成功")
    print("長度：", len(html))
    print("Citation：", document["citation"])
    print(f"結果已輸出到 {output_path}")


if __name__ == "__main__":
    main()
