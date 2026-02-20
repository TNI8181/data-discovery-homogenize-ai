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
# Homogenization (Safer / Less Overboard)
# 1) Hard grouping by "collapsed_key" (Policy Number == policy_number == policynumber)
# 2) Optional AI only to choose canonical WITHIN that group
# 3) Never groups unrelated fields
# =========================================================
def _collapsed_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _pick_best_existing_variant(variants):
    """
    Choose most business-readable label among existing variants.
    Preference: spaces + Title Case-ish, avoid underscores/dashes.
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
    if not variants:
        return str(llm_canonical).strip()

    canon_norm = str(llm_canonical).strip().lower()
    for v in variants:
        if str(v).strip().lower() == canon_norm:
            return v
    return _pick_best_existing_variant(variants)

def run_homogenization_conservative(unique_cols, api_key: str, llm_model: str, use_llm: bool):
    """
    Conservative grouping ONLY by collapsed_key equality.
    Returns:
      canonical_map[col] -> canonical (one of original variants)
      group_sizes[col]   -> size of group
      legacy_list_by_canon[canonical] -> "a, b, c" (excluding canonical itself)
      group_variants_by_canon[canonical] -> set(all variants, including canonical)
    """
    texts = [str(x) for x in unique_cols if str(x).strip() != ""]
    if not texts:
        return {}, {}, {}, {}

    key_to_group = {}
    for t in texts:
        key_to_group.setdefault(_collapsed_key(t), []).append(t)

    canonical_map = {}
    group_sizes = {}
    legacy_list_by_canon = {}
    group_variants_by_canon = {}

    for _, group in key_to_group.items():
        group = [str(x) for x in group]
        uniq_group = sorted(set(group), key=lambda x: (x.lower(), x))

        if len(uniq_group) == 1:
            canon = uniq_group[0]
        else:
            if use_llm:
                result = _pick_canonical_for_group_llm(group_variants=uniq_group, api_key=api_key, llm_model=llm_model)
                llm_canon = str(result.get("canonical", "")).strip()
                canon = _match_llm_canonical_to_existing(llm_canon, uniq_group)
            else:
                canon = _pick_best_existing_variant(uniq_group)

        canon_norm = canon.strip().lower()
        legacy_only = sorted([v for v in uniq_group if v.strip().lower() != canon_norm])
        legacy_list_by_canon[canon] = ", ".join(legacy_only)
        group_variants_by_canon[canon] = set(uniq_group)

        for v in uniq_group:
            canonical_map[v] = canon
            group_sizes[v] = len(uniq_group)

    return canonical_map, group_sizes, legacy_list_by_canon, group_variants_by_canon


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
    "Enable Homogenization (safe: groups only obvious variants like Policy Number vs Policy_number vs policynumber)",
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

    # =========================================================
    # RAW INVENTORY you asked for:
    # - file name is the report name
    # - do NOT append sheet name
    # - include ALL columns from ALL sheets (including Consolidated)
    # - if the same column appears in multiple sheets in the same file, we keep it once for that file
    # =========================================================
    st.write("## Raw Column Inventory (File = Report Name)")
    raw_rows = []
    for f in uploaded_files:
        try:
            report_name = f.name  # ✅ file name only
            if f.name.lower().endswith(".csv"):
                df = read_csv_flexible(f)
                for col in df.columns:
                    raw_rows.append({"report_name": report_name, "column_original": str(col)})
            else:
                xls = pd.ExcelFile(f)
                # collect unique columns across all sheets for this file
                cols_set = set()
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    cols_set.update([str(c) for c in df.columns])
                for col in sorted(cols_set, key=lambda x: (x.lower(), x)):
                    raw_rows.append({"report_name": report_name, "column_original": col})
        except Exception as e:
            st.error(f"Could not process {f.name}: {e}")

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        st.warning("No columns found in uploaded files.")
        st.stop()

    st.dataframe(raw_df, use_container_width=True, hide_index=True)

    # =========================================================
    # Homogenization maps built from ALL distinct columns across ALL reports
    # This ensures you don't "miss" variants like Policy_number / policynumber etc.
    # =========================================================
    canonical_map, group_sizes, legacy_list_by_canon, group_variants_by_canon = {}, {}, {}, {}
    if enable_homog:
        all_unique_cols = sorted(raw_df["column_original"].dropna().astype(str).unique().tolist(), key=lambda x: (x.lower(), x))
        with st.spinner("Homogenizing (safe grouping by collapsed key)..."):
            try:
                canonical_map, group_sizes, legacy_list_by_canon, group_variants_by_canon = run_homogenization_conservative(
                    unique_cols=all_unique_cols,
                    api_key=api_key,
                    llm_model=llm_model.strip(),
                    use_llm=bool(use_llm_for_canonical),
                )
            except Exception as e:
                st.error(f"Homogenization failed: {e}")
                canonical_map, group_sizes, legacy_list_by_canon, group_variants_by_canon = {}, {}, {}, {}

    # =========================================================
    # Cross Tab
    # - Columns = ONLY report names (file names)
    # - Rows:
    #    * If a group has >1 variants => show only canonical
    #    * Else show the column itself
    # - Cell = "x" if ANY variant for that canonical exists in that report
    # - legacy_columns = list of legacy variants under that canonical (excluding canonical)
    # - IMPORTANT: we still mark the report with x even if only policynumber exists there (rolled under Policy Number)
    # =========================================================
    st.write("## Report vs Field Cross Tab (X = Present)")

    # Build a working DF from raw_df (already file-level, sheet-agnostic)
    tmp = raw_df.copy()

    if enable_homog and canonical_map:
        tmp["canonical"] = tmp["column_original"].map(canonical_map).fillna(tmp["column_original"])
        tmp["group_size"] = tmp["column_original"].map(group_sizes).fillna(1).astype(int)
        tmp["row_field"] = np.where(tmp["group_size"] > 1, tmp["canonical"], tmp["column_original"])
    else:
        tmp["row_field"] = tmp["column_original"]

    # Presence: if ANY original column (canonical or any legacy) appears in a report, row_field is present
    collapsed = tmp.groupby(["row_field", "report_name"], as_index=False).size().drop(columns=["size"])
    cross_counts = pd.crosstab(collapsed["row_field"], collapsed["report_name"])
    cross_tab = cross_counts.applymap(lambda v: "x" if v > 0 else "")

    # legacy list column (TEXT)
    if enable_homog and legacy_list_by_canon:
        legacy_series = pd.Series(cross_tab.index, index=cross_tab.index).map(lambda k: legacy_list_by_canon.get(k, "")).fillna("")
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

# Reference file (for your internal tracking / the uploaded rtf)
# :contentReference[oaicite:0]{index=0}
