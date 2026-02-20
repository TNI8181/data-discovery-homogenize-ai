import streamlit as st
import pandas as pd
import io
import numpy as np
import json
import os
import re

# =========================================================
# Helper: Flexible CSV Reader
# =========================================================
def read_csv_flexible(uploaded_file):
    """
    Handles:
    - Normal CSV
    - CSV with BOM
    - CSV where each entire row is wrapped in quotes:
      "col1,col2,col3"
      "val1,val2,val3"
    """
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except Exception:
        df = None

    # If it parsed into a single column, likely "whole-row quoted" CSV
    if df is None or df.shape[1] == 1:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw

        fixed_lines = []
        for line in text.splitlines():
            line = line.strip()
            if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
                line = line[1:-1]
            fixed_lines.append(line)

        df = pd.read_csv(io.StringIO("\n".join(fixed_lines)))

    return df


# =========================================================
# Homogenization (NO normalization/case changes)
# - We NEVER modify a column name string.
# - We ONLY group variants by collapsed_key and pick a canonical FROM THE EXISTING VARIANTS.
# =========================================================
def _collapsed_key(s: str) -> str:
    # Grouping key only; never displayed.
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _pick_best_existing_variant(variants_in_order: list[str]) -> str:
    """
    Choose the most business-readable variant FROM the existing variants.
    Returns the exact original string (no casing changes).
    """
    def score(v: str) -> tuple:
        s = str(v).strip()
        has_space = " " in s
        has_underscore = "_" in s
        has_dash = "-" in s
        is_all_lower = s.islower()
        is_all_upper = s.isupper()

        words = [w for w in s.replace("_", " ").replace("-", " ").split() if w]
        titleish = sum(1 for w in words if w[:1].isupper()) >= max(1, int(0.7 * len(words)))
        short_token_penalty = sum(1 for w in words if len(w) <= 3)

        # Higher is better
        return (
            1 if has_space else 0,
            1 if titleish else 0,
            0 if not has_underscore else -1,
            0 if not has_dash else -1,
            1 if not is_all_lower else 0,
            1 if not is_all_upper else 0,
            -short_token_penalty,
            len(s),
        )

    if not variants_in_order:
        return ""

    # Stable tie-breaker: keep earlier-seen if scores equal
    scored = [(score(v), i, v) for i, v in enumerate(variants_in_order)]
    scored.sort(key=lambda t: (t[0], -t[1]))  # prefer higher score; earlier index wins via -i
    return scored[-1][2]

@st.cache_data(show_spinner=False)
def _pick_canonical_for_group_llm(group_variants_in_order: list[str], api_key: str, llm_model: str) -> dict:
    """
    LLM suggests canonical; we still FORCE canonical to be one of the group variants.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "canonical": {"type": "string"},
            "variants": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["canonical", "variants"],
    }

    prompt = f"""
You are helping homogenize legacy data field names for an insurance data discovery tool.

Given a list of column name variants that likely mean the same business concept:

1) Suggest the best canonical name for business users.
2) Canonical should be readable (e.g., "Policy Number" instead of "policynumber").
3) IMPORTANT: The canonical MUST be one of the provided variants (do not invent a new label).
4) Return the variants exactly as provided.

Variants:
{group_variants_in_order}
"""

    resp = client.responses.create(
        model=llm_model,
        input=prompt,
        text={"format": {"name": "homogenization_result", "type": "json_schema", "schema": schema}},
    )

    try:
        parsed = resp.output_parsed
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback
    try:
        return json.loads(resp.output_text)
    except Exception:
        return {"canonical": group_variants_in_order[0], "variants": group_variants_in_order}

def _match_llm_canonical_to_existing(llm_canonical: str, variants_in_order: list[str]) -> str:
    """
    Force canonical to be EXACTLY one of the existing variants (preserve original case).
    """
    if not variants_in_order:
        return str(llm_canonical).strip()

    canon_norm = str(llm_canonical).strip().lower()
    for v in variants_in_order:
        if str(v).strip().lower() == canon_norm:
            return v

    return _pick_best_existing_variant(variants_in_order)

def run_homogenization_conservative(all_unique_cols_in_order: list[str], api_key: str, llm_model: str, use_llm: bool):
    """
    Conservative grouping ONLY by collapsed_key equality.

    Returns:
      canonical_map[col] -> canonical (exact existing variant string)
      group_sizes[col]   -> group size
      legacy_list_by_canon[canonical] -> comma-separated legacy variants (exact strings)
    """
    # Build groups in encounter order
    key_to_variants = {}
    key_order = []
    for col in all_unique_cols_in_order:
        c = str(col)
        if not c.strip():
            continue
        k = _collapsed_key(c)
        if k not in key_to_variants:
            key_to_variants[k] = []
            key_order.append(k)
        if c not in key_to_variants[k]:
            key_to_variants[k].append(c)  # preserve original string + order

    canonical_map = {}
    group_sizes = {}
    legacy_list_by_canon = {}

    for k in key_order:
        variants = key_to_variants[k]
        if len(variants) == 1:
            canon = variants[0]
        else:
            if use_llm:
                result = _pick_canonical_for_group_llm(variants, api_key=api_key, llm_model=llm_model)
                llm_canon = str(result.get("canonical", "")).strip()
                canon = _match_llm_canonical_to_existing(llm_canon, variants)
            else:
                canon = _pick_best_existing_variant(variants)

        canon_norm = canon.strip().lower()
        legacy = [v for v in variants if v.strip().lower() != canon_norm]
        legacy_list_by_canon[canon] = ", ".join(legacy)

        for v in variants:
            canonical_map[v] = canon
            group_sizes[v] = len(variants)

    return canonical_map, group_sizes, legacy_list_by_canon


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Insurance Data Discovery", layout="wide")
st.title("Insurance Data Discovery Tool (Module 1)")
st.caption("Upload sample reports (Excel/CSV). We extract column metadata and build a report-vs-field cross-tab.")

source_system = st.text_input("Source System Name (e.g., Legacy PAS, Mainframe Claims)")

uploaded_files = st.file_uploader(
    "Upload report files",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

st.markdown("---")
st.subheader("Homogenization")

enable_homog = st.checkbox(
    "Enable Homogenization (groups only obvious variants like Policy Number / Policy_number / policynumber)",
    value=True
)

use_llm_for_canonical = st.checkbox(
    "Use AI to pick canonical name within each group (optional). If off, heuristic will pick canonical.",
    value=True
)

# Prefer secrets/env; allow UI override
api_key = (st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else "") or os.getenv("OPENAI_API_KEY", "")
api_key_ui = st.text_input("OpenAI API Key (optional if set in Secrets/env)", type="password")
api_key = (api_key_ui.strip() or api_key.strip())

llm_model = st.text_input("LLM model", value="gpt-4.1-mini")

analyze = st.button("Analyze Reports", type="primary")


# =========================================================
# Main Processing
# =========================================================
if analyze:
    if not uploaded_files:
        st.warning("Please upload at least one Excel or CSV report.")
        st.stop()

    if not source_system.strip():
        st.warning("Please enter a Source System Name.")
        st.stop()

    if enable_homog and use_llm_for_canonical and not api_key:
        st.warning("AI canonical selection is enabled. Please provide an OpenAI API key (or turn off AI canonical selection).")
        st.stop()

    # =========================================================
    # RAW INVENTORY (File = Report Name)
    # - File name is the report name
    # - Do NOT append sheet name
    # - Include ALL columns from ALL sheets (including Consolidated)
    # - De-dupe per file while preserving FIRST-SEEN order (no sorting, no case changes)
    # =========================================================
    st.write("## Raw Column Inventory (File = Report Name)")
    raw_rows = []

    for f in uploaded_files:
        report_name = f.name  # file name ONLY

        try:
            if f.name.lower().endswith(".csv"):
                df = read_csv_flexible(f)
                seen = set()
                for col in list(df.columns):
                    col_str = str(col)
                    if col_str not in seen:
                        raw_rows.append({"report_name": report_name, "column_original": col_str})
                        seen.add(col_str)

            else:
                xls = pd.ExcelFile(f)
                seen = set()
                # Sheet order is preserved as Excel provides it; we do NOT show sheet names.
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    for col in list(df.columns):
                        col_str = str(col)
                        if col_str not in seen:
                            raw_rows.append({"report_name": report_name, "column_original": col_str})
                            seen.add(col_str)

        except Exception as e:
            st.error(f"Could not process {f.name}: {e}")

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        st.warning("No columns found in uploaded files.")
        st.stop()

    st.dataframe(raw_df, use_container_width=True, hide_index=True)

    # =========================================================
    # Build homogenization maps from ALL unique columns across ALL reports
    # - Preserve first-seen order across the entire dataset
    # - NEVER change casing of any column name string
    # =========================================================
    canonical_map, group_sizes, legacy_list_by_canon = {}, {}, {}

    if enable_homog:
        all_unique_cols_in_order = []
        seen_global = set()
        for col in raw_df["column_original"].tolist():
            if col not in seen_global:
                all_unique_cols_in_order.append(col)
                seen_global.add(col)

        with st.spinner("Homogenizing (safe grouping by collapsed key; no case changes)..."):
            try:
                canonical_map, group_sizes, legacy_list_by_canon = run_homogenization_conservative(
                    all_unique_cols_in_order=all_unique_cols_in_order,
                    api_key=api_key,
                    llm_model=llm_model.strip(),
                    use_llm=bool(use_llm_for_canonical),
                )
            except Exception as e:
                st.error(f"Homogenization failed: {e}")
                canonical_map, group_sizes, legacy_list_by_canon = {}, {}, {}

    # =========================================================
    # Cross Tab
    # - Columns = ONLY report names (file names)
    # - Rows:
    #    * If a group has >1 variants => show only canonical (exact variant chosen from existing names)
    #    * Else show the column itself (exact as seen)
    # - Cell = "x" if ANY variant under that canonical exists in the report
    # - legacy_columns = list of legacy variants under that canonical (exact strings; no case changes)
    # - No visible 0-based row index (reset_index + hide_index)
    # =========================================================
    st.write("## Report vs Field Cross Tab (X = Present)")

    tmp = raw_df.copy()

    if enable_homog and canonical_map:
        tmp["canonical"] = tmp["column_original"].map(canonical_map).fillna(tmp["column_original"])
        tmp["group_size"] = tmp["column_original"].map(group_sizes).fillna(1).astype(int)
        tmp["row_field"] = np.where(tmp["group_size"] > 1, tmp["canonical"], tmp["column_original"])
    else:
        tmp["row_field"] = tmp["column_original"]

    # Presence: any original variant implies the canonical row is present in that report
    collapsed = (
        tmp.groupby(["row_field", "report_name"], as_index=False)
           .size()
           .drop(columns=["size"])
    )

    cross_counts = pd.crosstab(collapsed["row_field"], collapsed["report_name"])
    cross_tab = cross_counts.applymap(lambda v: "x" if v > 0 else "")

    # Add legacy_columns TEXT (exact strings)
    if enable_homog and legacy_list_by_canon:
        legacy_series = (
            pd.Series(cross_tab.index, index=cross_tab.index)
              .map(lambda k: legacy_list_by_canon.get(k, ""))
              .fillna("")
        )
    else:
        legacy_series = pd.Series([""] * len(cross_tab.index), index=cross_tab.index)

    cross_tab.insert(0, "legacy_columns", legacy_series)

    # Repetition Count across report columns only
    report_cols = [c for c in cross_tab.columns if c != "legacy_columns"]
    cross_tab["Repetition Count"] = (cross_tab[report_cols] == "x").sum(axis=1)

    # Totals row
    totals = (cross_tab[report_cols] == "x").sum(axis=0)
    totals["legacy_columns"] = ""
    totals["Repetition Count"] = int(cross_tab["Repetition Count"].sum())
    cross_tab.loc["Totals"] = totals

    # Remove 0-based visible index by resetting and hiding index
    cross_tab = cross_tab.reset_index().rename(columns={"row_field": "column_original"})
    st.dataframe(cross_tab, use_container_width=True, hide_index=True)
