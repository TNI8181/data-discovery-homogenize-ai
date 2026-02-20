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
# Homogenization (Conservative)
# - Primary grouping rule: SAME collapsed_key => same group
# - This avoids over-grouping and fixes incorrect cross-tabs
# - LLM ONLY picks the canonical WITHIN each group (no invented labels)
# =========================================================
def _collapsed_key(s: str) -> str:
    """
    Very conservative grouping key:
    - lower
    - remove anything that's not a-z or 0-9
    Examples:
      "Policy Number" == "policynumber" == "Policy_Number"
    """
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _pick_best_existing_variant(variants):
    """
    Choose the most business-readable variant FROM the existing variants.
    Preference: spaces + Title Case-ish, avoid underscores/dashes, avoid all-lower/all-upper.
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

        return (
            1 if has_space else 0,
            1 if titleish else 0,
            0 if not has_underscore else -1,
            0 if not has_dash else -1,
            1 if not is_all_lower else 0,
            1 if not is_all_upper else 0,
            -short_token_penalty,
            len(s)
        )

    return sorted(list(variants), key=score, reverse=True)[0] if variants else ""

@st.cache_data(show_spinner=False)
def _pick_canonical_for_group_llm(group_variants, api_key: str, llm_model: str):
    """
    LLM suggests canonical; we still FORCE canonical to be one of the group_variants.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "canonical": {"type": "string"},
            "variants": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["canonical", "variants"]
    }

    prompt = f"""
You are helping homogenize legacy data field names for an insurance data discovery tool.

Given a list of column name variants that likely mean the same business concept:

1) Suggest the best canonical name for business users.
2) Canonical should be Title Case with spaces (NOT snake_case).
3) Avoid abbreviations if a clearer term exists.
4) Return the variants exactly as provided.

Variants:
{group_variants}
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

    try:
        return json.loads(resp.output_text)
    except Exception:
        return {"canonical": group_variants[0], "variants": group_variants}

def _match_llm_canonical_to_existing(llm_canonical: str, variants):
    """
    Force canonical to be one of the existing variants.
    """
    if not variants:
        return str(llm_canonical).strip()

    canon_norm = str(llm_canonical).strip().lower()
    for v in variants:
        if str(v).strip().lower() == canon_norm:
            return v
    return _pick_best_existing_variant(variants)

def run_homogenization_conservative(unique_cols, api_key: str, llm_model: str, use_llm: bool):
    """
    Conservative grouping: ONLY collapsed_key equality.
    Returns:
      canonical_map[col] -> canonical (one of original variants)
      group_sizes[col]   -> size of group
      legacy_list_by_canon[canonical] -> "a, b, c" (excluding canonical itself)
    """
    texts = list(unique_cols)
    if not texts:
        return {}, {}, {}

    # Group by collapsed_key
    key_to_group = {}
    for t in texts:
        key_to_group.setdefault(_collapsed_key(t), []).append(t)

    canonical_map = {}
    group_sizes = {}
    legacy_list_by_canon = {}

    for _, group in key_to_group.items():
        group = [str(x) for x in group]
        if len(group) == 1:
            canon = group[0]
        else:
            if use_llm:
                result = _pick_canonical_for_group_llm(group_variants=group, api_key=api_key, llm_model=llm_model)
                llm_canon = str(result.get("canonical", "")).strip()
                canon = _match_llm_canonical_to_existing(llm_canon, group)
            else:
                canon = _pick_best_existing_variant(group)

        # legacy list for this canonical (exclude canonical itself, case-insensitive)
        canon_norm = canon.strip().lower()
        legacy_only = sorted([v for v in set(group) if v.strip().lower() != canon_norm])
        legacy_list_by_canon[canon] = ", ".join(legacy_only)

        for v in group:
            canonical_map[v] = canon
            group_sizes[v] = len(group)

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
    "Enable Homogenization (conservative: groups only obvious variants like Policy Number vs policy_number vs policynumber)",
    value=True
)

use_llm_for_canonical = st.checkbox(
    "Use AI to pick canonical name within each group (recommended). If off, heuristic will pick canonical.",
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

    # -------------------------------
    # Quick Profiling (includes ALL sheets, including Consolidated)
    # -------------------------------
    st.write("## Quick Profiling (Preview)")
    profile_rows = []
    for f in uploaded_files:
        try:
            if f.name.lower().endswith(".csv"):
                df = read_csv_flexible(f)
                profile_rows.append({
                    "report_name": f.name,
                    "rows": int(len(df)),
                    "columns": int(len(df.columns)),
                    "sample_columns": ", ".join([str(c) for c in df.columns[:12]])
                })
            else:
                xls = pd.ExcelFile(f)
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    profile_rows.append({
                        "report_name": f"{f.name} | {sheet}",  # ✅ includes Consolidated explicitly
                        "rows": int(len(df)),
                        "columns": int(len(df.columns)),
                        "sample_columns": ", ".join([str(c) for c in df.columns[:12]])
                    })
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")

    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    # -------------------------------
    # Build Field Inventory (Raw) (includes ALL sheets)
    # -------------------------------
    field_rows = []
    for f in uploaded_files:
        try:
            if f.name.lower().endswith(".csv"):
                df = read_csv_flexible(f)
                report_label = f.name
                for col in df.columns:
                    field_rows.append({"report_name": report_label, "column_original": str(col)})
            else:
                xls = pd.ExcelFile(f)
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    report_label = f"{f.name} | {sheet}"  # ✅ includes Consolidated explicitly
                    for col in df.columns:
                        field_rows.append({"report_name": report_label, "column_original": str(col)})
        except Exception as e:
            st.error(f"Could not process {f.name}: {e}")

    field_df = pd.DataFrame(field_rows)
    if field_df.empty:
        st.warning("No columns found in uploaded files.")
        st.stop()

    # -------------------------------
    # Homogenization maps (conservative)
    # -------------------------------
    canonical_map = {}
    group_sizes = {}
    legacy_list_by_canon = {}

    if enable_homog:
        unique_cols = sorted(field_df["column_original"].dropna().astype(str).unique().tolist())
        with st.spinner("Homogenizing (conservative grouping)..."):
            try:
                canonical_map, group_sizes, legacy_list_by_canon = run_homogenization_conservative(
                    unique_cols=unique_cols,
                    api_key=api_key,
                    llm_model=llm_model.strip(),
                    use_llm=bool(use_llm_for_canonical),
                )
            except Exception as e:
                st.error(f"Homogenization failed: {e}")
                canonical_map, group_sizes, legacy_list_by_canon = {}, {}, {}

    # =========================================================
    # Cross Tab
    #
    # ✅ Correctness goals:
    # - Columns = ONLY report names (sheet-aware, includes Consolidated)
    # - Rows:
    #    * If a group has >1 variants => show only the chosen canonical as the row (column_original)
    #    * Else show the column itself
    # - Cell = "x" if ANY variant belonging to that canonical group exists in that report
    # - legacy_columns = text list of the variants that were rolled up under that canonical (excluding the canonical itself)
    # - No row numbers starting at 0: we output index as a real column (reset_index) and hide pandas index
    # =========================================================
    st.write("## Report vs Field Cross Tab (X = Present)")

    tmp = field_df.copy()
    if enable_homog and canonical_map:
        tmp["canonical"] = tmp["column_original"].map(canonical_map).fillna(tmp["column_original"])
        tmp["group_size"] = tmp["column_original"].map(group_sizes).fillna(1).astype(int)
        tmp["row_field"] = np.where(tmp["group_size"] > 1, tmp["canonical"], tmp["column_original"])
    else:
        tmp["row_field"] = tmp["column_original"]

    # Presence across variants: one row per (row_field, report_name) if any variant exists
    collapsed = tmp.groupby(["row_field", "report_name"], as_index=False).size().drop(columns=["size"])
    cross_counts = pd.crosstab(collapsed["row_field"], collapsed["report_name"])
    cross_tab = cross_counts.applymap(lambda v: "x" if v > 0 else "")

    # Insert legacy list column (TEXT)
    if enable_homog and legacy_list_by_canon:
        legacy_series = pd.Series(cross_tab.index, index=cross_tab.index).map(lambda k: legacy_list_by_canon.get(k, "")).fillna("")
    else:
        legacy_series = pd.Series([""] * len(cross_tab.index), index=cross_tab.index)

    cross_tab.insert(0, "legacy_columns", legacy_series)

    # Repetition Count (report columns only)
    report_cols = [c for c in cross_tab.columns if c != "legacy_columns"]
    cross_tab["Repetition Count"] = (cross_tab[report_cols] == "x").sum(axis=1)

    # Totals row
    totals = (cross_tab[report_cols] == "x").sum(axis=0)
    totals["legacy_columns"] = ""
    totals["Repetition Count"] = int(cross_tab["Repetition Count"].sum())
    cross_tab.loc["Totals"] = totals

    # Make row_field a real column to avoid 0-based row index display
    cross_tab = cross_tab.reset_index().rename(columns={"row_field": "column_original"})

    st.dataframe(cross_tab, use_container_width=True, hide_index=True)
