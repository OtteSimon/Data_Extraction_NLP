import re

import pdfplumber
from googletrans import Translator
from langdetect import detect


SECTION_1_TO_3_PATTERNS = [
    (r"ABSCHNITT 1:", r"ABSCHNITT 4:"),
    (r"SECTION 1:", r"SECTION 4:"),
    (r"SECTION 1\.", r"SECTION 4\."),
    (r"SECTION 1 - ", r"SECTION 4 - "),
    (
        r"SECTION 1 IDENTIFICATION OF THE SUBSTANCE/MIXTURE AND OF THE COMPANY ",
        r" SECTION 4 FIRST AID MEASURES",
    ),
    (
        r"1\. Identification of the substance/mixture and of the company/undertaking",
        r"4\. First aid measures",
    ),
    (
        r"1\. Bezeichnung des Stoffs bzw. des Gemischs und des Unternehmens",
        r"4\. Erste-Hilfe-Maßnahmen",
    ),
    (r"1\. Bezeichnung des Stoffes und des Unternehmens", r"4\. Erste-Hilfe-Maßnahmen"),
    (r"1\. Bezeichnung des Stoffs und des Unternehmens", r"4\. Erste-Hilfe-Maßnahmen"),
    (r"1\. PRODUCT AND COMPANY IDENTIFICATION", r"4\. FIRST AID MEASURES"),
    (r"1\. IDENTIFICATION", r"4\. FIRST-AID MEASURES"),
    (r"1 Bezeichnung des Stoffes bzw. Gemisches", r"4 Erste Hilfe"),
    (r"1 Identification of the Substance", r"4 First aid measures"),
    (r"1 - CHEMICAL PRODUCT AND COMPANY IDENTIFICATION", r"4 - FIRST-AID MEASURES"),
    (r"1 - CHEMICAL PRODUCT AND COMPANY IDENTIFICATION", r"4 - FIRST AID MEASURES"),
]

SECTION_9_PATTERNS = [
    (r"ABSCHNITT 9:", r"ABSCHNITT 10:"),
    (r"SECTION 9:", r"SECTION 10:"),
    (r"SECTION 9\.", r"SECTION 10\."),
    (r"SECTION 9 -", r"SECTION 10 -"),
    (r"SECTION 9 PHYSICAL AND CHEMICAL PROPERTIES", r"SECTION 10 STABILITY AND REACTIVITY"),
    (r"9\. Physikalische und chemische Eigenschaften", r"10\. Stabilität und Reaktivität"),
    (r"9\. PHYSICAL AND CHEMICAL PROPERTIES", r"10\. STABILITY AND REACTIVITY"),
    (r"9 - PHYSICAL AND CHEMICAL PROPERTIES", r"10 - STABILITY AND REACTIVITY"),
    (r"9 Physical and chemical properties", r"10 Stability and reactivity"),
    (r"9 physikalische/chemische Eigenschaften", r"10 Stabilität und Reaktivität"),
    (
        r"9\.1 Information on basic physical and chemical properties",
        r"Section 10: Stability and reactivity",
    ),
]


def extract_and_preprocess(pdf_path):
    if count_pages(pdf_path=pdf_path) > 2:
        text = extract_text_without_header_footer(pdf_path=pdf_path)
        text = extract_relevant_sections(text=text)
        is_material_data_sheet = False
    else:
        text = extract_text(pdf_path=pdf_path)
        is_material_data_sheet = True

    if detect_language(text) == "en":
        text = translate_to_german(text)

    return normalize_text(text=text), is_material_data_sheet


def extract_relevant_sections(text):
    section_1_to_3 = extract_section(text, SECTION_1_TO_3_PATTERNS)
    section_9 = extract_section(text, SECTION_9_PATTERNS)
    return section_1_to_3 + " " + section_9


def extract_section(text, patterns):
    for start_pattern, end_pattern in patterns:
        regex = rf"(?i){start_pattern}(.*?){end_pattern}"
        match = re.search(regex, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return "No matching section found."


def count_pages(pdf_path):
    with pdfplumber.open(pdf_path) as pdf_file:
        return len(pdf_file.pages)


def extract_text(pdf_path):
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf_file:
        for page in pdf_file.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=4) or ""
            extracted_text += page_text + " \n"
    return extracted_text


def extract_text_without_header_footer(pdf_path):
    with pdfplumber.open(pdf_path) as pdf_file:
        pages = pdf_file.pages
        if len(pages) < 3:
            return extract_text(pdf_path)

        second_page_text = pages[1].extract_text(x_tolerance=2, y_tolerance=4) or ""
        third_page_text = pages[2].extract_text(x_tolerance=2, y_tolerance=4) or ""

        header_start, header_end = find_repeating_lines(
            second_page_text,
            third_page_text,
            from_top=True,
        )
        footer_start, footer_end = find_repeating_lines(
            second_page_text,
            third_page_text,
            from_top=False,
        )

        full_text = ""
        for page in pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=4) or ""
            page_lines = page_text.split("\n")

            if header_start is not None and header_end is not None:
                del page_lines[header_start:header_end]

            if footer_start is not None and footer_end is not None:
                for _ in range(footer_end + 1):
                    if page_lines:
                        del page_lines[-1]

            full_text += " \n".join(page_lines) + " \n"

        return full_text.strip()


def find_repeating_lines(text_1, text_2, from_top=True):
    start_index = None
    end_index = None
    lines_1 = text_1.split("\n")
    lines_2 = text_2.split("\n")
    max_lines_to_check = min(len(lines_1), len(lines_2), 5)

    if not from_top:
        lines_1 = lines_1[::-1]
        lines_2 = lines_2[::-1]

    for index in range(max_lines_to_check):
        clean_line_1 = re.sub(r"\d+", "", lines_1[index]).strip()
        clean_line_2 = re.sub(r"\d+", "", lines_2[index]).strip()
        if clean_line_1 == clean_line_2:
            if start_index is None:
                start_index = index
            end_index = index
        elif start_index is not None:
            break

    return start_index, end_index


def normalize_text(text):
    text = re.sub(r"<.*?>", "", text)
    return " ".join(text.split())


def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "error"


def translate_to_german(text):
    translator = Translator()
    translated = translator.translate(text=text, src="en", dest="de")
    return translated.text
