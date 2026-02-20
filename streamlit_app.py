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
# AI Homogenization: Embeddings + Similarity Grouping + Canonical Selection
# =========================================================
def _cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v = vectors / norms
    return v @ v.T

def _collapsed_key(s: str) -> str:
    """
    Strong grouping key:
    - lower
    - remove anything that's not a-z or 0-9
    Example: "Policy Number" == "policynumber" == "Policy_Number"
    """
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _union_find_groups(sim_mat: np.ndarray, threshold: float, texts: list[str]):
    """
    Groups columns by:
    (A) Strong rule: same collapsed_key => same group
    (B) Embedding similarity: cosine >= threshold => same group
    """
    n = sim_mat.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # (A) Strong rule unions
    key_to_idxs = {}
    for i, t in enumerate(texts):
        key_to_idxs.setdefault(_collapsed_key(t), []).append(i)
    for idxs in key_to_idxs.values():
        if len(idxs) > 1:
            root = idxs[0]
            for j in idxs[1:]:
                union(root, j)

    # (B) Similarity unions
    for i in range(n):
        for j in range(i + 1, n):
            if sim_mat[i, j] >= threshold:
                union(i, j)

    # Collect groups
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())

@st.cache_data(show_spinner=False)
def _embed_texts(texts, api_key: str, embedding_model: str):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing dependency: openai. Add `openai>=1.0.0` to requirements.txt") from e

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=embedding_model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=float)

@st.cache_data(show_spinner=False)
def _pick_canonical_for_group(group_variants, api_key: str, llm_model: str):
    """
    LLM suggests a business-readable canonical label for a group.
    We'll force canonical to be one of the existing variants (no invented labels).
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

    response = client.responses.create(
        model=llm_model,
        input=prompt,
        text={"format": {"name": "homogenization_result", "type": "json_schema", "schema": schema}},
    )

    try:
        parsed = response.output_parsed
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        return json.loads(response.output_text)
    except Exception:
        return {"canonical": group_variants[0], "variants": group_variants}

def _pick_best_existing_variant(variants):
    """
    Pick the most business-readable variant FROM the existing variants (no synonym dictionaries).
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

def run_homogenization(unique_cols, api_key: str, embedding_model: str, llm_model: str, threshold: float):
    """
    Returns:
      canonical_map[col] -> canonical (one of the original variants)
      variants_map[col]  -> comma-separated variants in its group (original text)
      group_sizes[col]   -> size of group
    """
    texts = list(unique_cols)
    if not texts:
        return {}, {}, {}

    vectors = _embed_texts(texts, api_key=api_key, embedding_model=embedding_model)
    sim = _cosine_sim_matrix(vectors)
    groups_idx = _union_find_groups(sim, threshold=threshold, texts=texts)

    canonical_map, variants_map, group_sizes = {}, {}, {}

    for idxs in groups_idx:
        group = [texts[i] for i in idxs]

        if len(group) == 1:
            canonical = group[0]
            variants = group
        else:
            result = _pick_canonical_for_group(group_variants=group, api_key=api_key, llm_model=llm_model)
            llm_canon = str(result.get("canonical", "")).strip()
            canonical = _match_llm_canonical_to_existing(llm_canon, group)
            variants = group

        variants_str = ", ".join([str(v) for v in variants])

        for v in group:
            canonical_map[v] = canonical
            variants_map[v] = variants_str
            group_sizes[v] = len(group)

    return canonical_map, variants_map, group_sizes


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
st.subheader("Homogenization (AI)")

enable_homog = st.checkbox(
    "Enable AI Homogenization (Policy Number + policynumber => keep best, roll variants into legacy_columns)",
    value=True
)

# Prefer secrets/env; allow UI override
api_key = (st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else "") or os.getenv("OPENAI_API_KEY", "")
api_key_ui = st.text_input("OpenAI API Key (optional if set in Secrets/env)", type="password")
api_key = (api_key_ui.strip() or api_key.strip())

c1, c2, c3 = st.columns(3)
with c1:
    embedding_model = st.text_input("Embedding model", value="text-embedding-3-small")
with c2:
    llm_model = st.text_input("LLM model", value="gpt-4.1-mini")
with c3:
    similarity_threshold = st.slider(
        "Similarity threshold (higher = stricter grouping)",
        min_value=0.70,
        max_value=0.95,
        value=0.84,
        step=0.01
    )

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

    if enable_homog and not api_key:
        st.warning("Homogenization is enabled. Please provide an OpenAI API key (or disable homogenization).")
        st.stop()

    # -------------------------------
    # Quick Profiling
    # -------------------------------
    st.write("## Quick Profiling (Preview)")
    profile_rows = []
    for f in uploaded_files:
        try:
            if f.name.lower().endswith(".csv"):
                df = read_csv_flexible(f)
                profile_rows.append({
                    "file_name": f.name,
                    "sheet_name": "(csv)",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "sample_columns": ", ".join([str(c) for c in df.columns[:10]])
                })
            else:
                xls = pd.ExcelFile(f)
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    profile_rows.append({
                        "file_name": f.name,
                        "sheet_name": sheet,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "sample_columns": ", ".join([str(c) for c in df.columns[:10]])
                    })
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")

    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    # -------------------------------
    # Build Field Inventory (Raw)
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
                report_label = f.name
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    for col in df.columns:
                        field_rows.append({"report_name": report_label, "column_original": str(col)})
        except Exception as e:
            st.error(f"Could not process {f.name}: {e}")

    field_df = pd.DataFrame(field_rows)

    # -------------------------------
    # AI Homogenization: Build maps
    # -------------------------------
    canonical_map, variants_map, group_sizes = {}, {}, {}
    if enable_homog and not field_df.empty:
        unique_cols = sorted(field_df["column_original"].dropna().astype(str).unique().tolist())
        with st.spinner("Running AI homogenization (strong rule + embeddings + canonical selection)..."):
            try:
                canonical_map, variants_map, group_sizes = run_homogenization(
                    unique_cols=unique_cols,
                    api_key=api_key,
                    embedding_model=embedding_model.strip(),
                    llm_model=llm_model.strip(),
                    threshold=float(similarity_threshold),
                )
            except Exception as e:
                st.error(f"Homogenization failed: {e}")
                canonical_map, variants_map, group_sizes = {}, {}, {}

    # =========================================================
    # Cross Tab (Report columns only) + legacy_columns list
    #
    # Requirements implemented:
    # - Columns = only report names
    # - Rows = canonical name for a homogenized group, else original field
    # - Cell = "x" if ANY variant (canonical or legacy) appears in that report
    # - legacy_columns (TEXT) = list of legacy variants for that canonical (comma-separated)
    # - Repetition Count = count of x across report columns
    # - Totals row included
    # =========================================================
    st.write("## Report vs Field Cross Tab (X = Present)")

    if not field_df.empty and enable_homog and canonical_map:
        tmp = field_df.copy()
        tmp["canonical"] = tmp["column_original"].map(canonical_map).fillna(tmp["column_original"])
        tmp["group_size"] = tmp["column_original"].map(group_sizes).fillna(1).astype(int)
        tmp["row_field"] = np.where(tmp["group_size"] > 1, tmp["canonical"], tmp["column_original"])

        # For each canonical, build full variant set (as found in dataset) and then legacy list (exclude canonical itself)
        canonical_to_variants = {}
        for _, r in tmp.iterrows():
            if int(r["group_size"]) <= 1:
                continue
            canon = str(r["canonical"])
            original = str(r["column_original"])
            canonical_to_variants.setdefault(canon, set()).add(original)

        canonical_to_legacy_list = {}
        for canon, varset in canonical_to_variants.items():
            canon_norm = canon.strip().lower()
            legacy = sorted([v for v in varset if v.strip().lower() != canon_norm])
            canonical_to_legacy_list[canon] = ", ".join(legacy)

        # Presence across variants: collapse to (row_field, report_name)
        collapsed = tmp.groupby(["row_field", "report_name"], as_index=False).size().drop(columns=["size"])

        cross_counts = pd.crosstab(collapsed["row_field"], collapsed["report_name"])
        cross_tab = cross_counts.applymap(lambda v: "x" if v > 0 else "")

        # Insert legacy list column (TEXT, not "x")
        cross_tab.insert(
            loc=0,
            column="legacy_columns",
            value=pd.Series(cross_tab.index, index=cross_tab.index).map(lambda k: canonical_to_legacy_list.get(k, "")).fillna("")
        )

        # Repetition Count across report columns only
        report_cols = [c for c in cross_tab.columns if c not in ["legacy_columns", "Repetition Count"]]
        cross_tab["Repetition Count"] = (cross_tab[report_cols] == "x").sum(axis=1)

        # Totals row
        totals = (cross_tab[report_cols] == "x").sum(axis=0)
        totals["legacy_columns"] = ""
        totals["Repetition Count"] = int(cross_tab["Repetition Count"].sum())
        cross_tab.loc["Totals"] = totals

        cross_tab.index.name = "column_original"
        st.dataframe(cross_tab, use_container_width=True, hide_index=False)

    else:
        # Fallback: raw crosstab
        cross_counts = pd.crosstab(field_df["column_original"], field_df["report_name"])
        cross_tab = cross_counts.applymap(lambda v: "x" if v > 0 else "")
        cross_tab["Repetition Count"] = (cross_tab == "x").sum(axis=1)

        totals = (cross_tab == "x").sum(axis=0)
        totals["Repetition Count"] = int(cross_tab["Repetition Count"].sum())
        cross_tab.loc["Totals"] = totals

        cross_tab.index.name = "column_original"
        st.dataframe(cross_tab, use_container_width=True, hide_index=False)
