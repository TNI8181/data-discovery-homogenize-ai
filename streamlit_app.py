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
    Returns: list of groups (each group is list of indices)
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

    # (A) Strong rule union
    key_to_idxs = {}
    for i, t in enumerate(texts):
        k = _collapsed_key(t)
        key_to_idxs.setdefault(k, []).append(i)

    for idxs in key_to_idxs.values():
        if len(idxs) > 1:
            root = idxs[0]
            for j in idxs[1:]:
                union(root, j)

    # (B) Similarity union
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
    """
    Returns embeddings as a numpy array in same order as texts.
    Cached by (texts, embedding_model). api_key included only for cache signature stability.
    """
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
    We'll later force canonical to be one of the existing variants (no invented labels).
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
        text={
            "format": {
                "name": "homogenization_result",
                "type": "json_schema",
                "schema": schema
            }
        }
    )

    # Prefer structured parse; fallback to JSON text
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
    Choose the most business-readable variant FROM the existing variants.
    Heuristic (no synonym dictionaries).
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
    - If LLM canonical matches an existing variant (case-insensitive), return that exact variant.
    - Else return best existing variant by heuristic.
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
      canonical_map[col] -> canonical (must be one of the original variants)
      variants_map[col]  -> comma-separated variants in its group (original text)
      group_sizes[col]   -> size of group
    """
    texts = list(unique_cols)
    if not texts:
        return {}, {}, {}

    vectors = _embed_texts(texts, api_key=api_key, embedding_model=embedding_model)
    sim = _cosine_sim_matrix(vectors)
    groups_idx = _union_find_groups(sim, threshold=threshold, texts=texts)

    canonical_map = {}
    variants_map = {}
    group_sizes = {}

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
st.caption(
    "Upload sample reports (Excel/CSV). We extract column metadata, build a field inventory, "
    "and produce a cross-tab. Optional AI homogenization groups similar fields and moves variants into legacy_columns."
)

source_system = st.text_input("Source System Name (e.g., Legacy PAS, Mainframe Claims)")

uploaded_files = st.file_uploader(
    "Upload report files",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

st.markdown("---")
st.subheader("Homogenization (AI)")

enable_homog = st.checkbox(
    "Enable AI Homogenization (Policy Number + policynumber => keep best in column_original, move others to legacy_columns)",
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

    st.success(f"Uploaded {len(uploaded_files)} file(s) for Source System: {source_system}")

    # -------------------------------
    # Show Uploaded Files
    # -------------------------------
    st.write("## Uploaded Files")
    for f in uploaded_files:
        st.write(f"• {f.name}")

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

    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

    # -------------------------------
    # Field Inventory (Raw)
    # -------------------------------
    st.write("## Field Inventory (Raw)")
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
                    report_label = f"{f.name}"
                    for col in df.columns:
                        field_rows.append({"report_name": report_label, "column_original": str(col)})
        except Exception as e:
            st.error(f"Could not process {f.name}: {e}")

    field_df = pd.DataFrame(field_rows)
    st.dataframe(field_df, use_container_width=True)

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

  # ================================
# Cross Tab (Canonical + Legacy cross-tab per report)
# WHAT CHANGED:
# - We STILL keep ONE row per business field (canonical shown as column_original)
# - BUT we ALSO "cross-tab" legacy usage per report:
#     For each report you will now see TWO columns:
#       1) <Report Name>                -> "x" if the canonical itself is present in that report
#       2) <Report Name> (Legacy Used)  -> comma-separated legacy variants that appeared in that report
# - This way, legacy columns are NOT ignored — you can see exactly where they occur.
# ================================
st.write("## Report vs Field Cross Tab (Canonical + Legacy Usage)")

if not field_df.empty:
    if enable_homog and canonical_map:
        tmp = field_df.copy()

        # Map each original to its canonical (canonical is one of the originals)
        tmp["canonical"] = tmp["column_original"].map(canonical_map).fillna(tmp["column_original"])
        tmp["group_size"] = tmp["column_original"].map(group_sizes).fillna(1).astype(int)

        # Display field (row label): canonical for groups>1, else original itself
        tmp["display_field"] = np.where(tmp["group_size"] > 1, tmp["canonical"], tmp["column_original"])

        # Build overall legacy list per canonical (for the legacy_columns column on the left)
        canonical_to_variants = {}
        for orig in tmp["column_original"].dropna().astype(str).unique():
            size = group_sizes.get(orig, 1)
            if size <= 1:
                continue
            canon = canonical_map.get(orig, orig)
            v_str = variants_map.get(orig, orig)
            parts = [p.strip() for p in str(v_str).split(",") if p.strip()]
            canonical_to_variants.setdefault(canon, set()).update(parts)

        canonical_to_legacy = {}
        for canon, varset in canonical_to_variants.items():
            canon_norm = str(canon).strip().lower()
            legacy_only = sorted([v for v in varset if str(v).strip().lower() != canon_norm])
            canonical_to_legacy[canon] = ", ".join(legacy_only)

        # For singletons, legacy_columns should be blank
        # (because there's nothing to homogenize)
        def overall_legacy_for_display(display_field: str) -> str:
            return canonical_to_legacy.get(display_field, "")

        # ---- Per report x legacy usage aggregation ----
        # For each (report_name, display_field) compute:
        # - canonical_present: did the canonical itself appear in this report?
        # - legacy_used: which legacy variants appeared in this report?
        def agg_group(g: pd.DataFrame) -> pd.Series:
            display = g["display_field"].iloc[0]
            # If group_size==1, treat as canonical
            if int(g["group_size"].iloc[0]) <= 1:
                return pd.Series({"canonical_present": True, "legacy_used": ""})

            canon = g["canonical"].iloc[0]
            canon_norm = str(canon).strip().lower()

            canon_present = any(str(x).strip().lower() == canon_norm for x in g["column_original"].tolist())

            legacy_hits = sorted({
                str(x).strip()
                for x in g["column_original"].tolist()
                if str(x).strip().lower() != canon_norm
            })

            return pd.Series({
                "canonical_present": bool(canon_present),
                "legacy_used": ", ".join(legacy_hits)
            })

        agg = (
            tmp.groupby(["report_name", "display_field"], as_index=False)
               .apply(agg_group)
               .reset_index(drop=True)
        )

        # Build a wide table with TWO columns per report
        reports = sorted(agg["report_name"].unique().tolist())
        fields = sorted(agg["display_field"].unique().tolist())

        # Start base table indexed by display_field (this is what user sees as column_original)
        cross_tab = pd.DataFrame(index=fields)
        cross_tab.index.name = "column_original"

        # Add overall legacy_columns next to column_original
        cross_tab.insert(
            loc=0,
            column="legacy_columns",
            value=pd.Series(fields, index=fields).map(overall_legacy_for_display).fillna("")
        )

        # Fill report columns
        for r in reports:
            sub = agg[agg["report_name"] == r].set_index("display_field")

            # Canonical X column
            cross_tab[r]
