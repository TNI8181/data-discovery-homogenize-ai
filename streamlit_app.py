import streamlit as st
import pandas as pd
import io
import numpy as np
import json
import os

# -------------------------------
# Helper: Flexible CSV Reader
# -------------------------------
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

    # Try normal read first
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except Exception:
        df = None

    # If only 1 column detected, likely wrapped-row CSV
    if df is None or df.shape[1] == 1:
        uploaded_file.seek(0)
        raw = uploaded_file.read()

        if isinstance(raw, bytes):
            text = raw.decode("utf-8-sig", errors="replace")
        else:
            text = raw

        fixed_lines = []
        for line in text.splitlines():
            line = line.strip()
            if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
                line = line[1:-1]
            fixed_lines.append(line)

        fixed_text = "\n".join(fixed_lines)
        df = pd.read_csv(io.StringIO(fixed_text))

    return df


# -------------------------------
# AI Homogenization (Embeddings + Similarity Grouping + Canonical Name via LLM)
# -------------------------------
def _cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v = vectors / norms
    return v @ v.T

def _union_find_groups(sim_mat: np.ndarray, threshold: float):
    """
    Simple grouping: if similarity(i,j) >= threshold => same group (connected components).
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

    for i in range(n):
        for j in range(i + 1, n):
            if sim_mat[i, j] >= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())

@st.cache_data(show_spinner=False)
def _embed_texts(texts, api_key: str, embedding_model: str):
    """
    Returns embeddings as a numpy array in same order as texts.
    Cached by (texts, embedding_model). api_key is included only to satisfy Streamlit cache signature.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing dependency: openai. Add `openai>=1.0.0` to requirements.txt") from e

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=embedding_model, input=texts)
    vectors = np.array([d.embedding for d in resp.data], dtype=float)
    return vectors

@st.cache_data(show_spinner=False)
def _pick_canonical_for_group(group_variants, api_key: str, llm_model: str):
    """
    Uses an LLM to pick the most business-readable canonical name (Option 1).
    Returns: {"canonical": "...", "variants": [...]}
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

1) Choose the best canonical name for business users.
2) Canonical must be Title Case with spaces (NOT snake_case).
3) Avoid abbreviations if a clearer term exists.
4) Return variants exactly as provided.

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
        if isinstance(parsed, dict) and "canonical" in parsed and "variants" in parsed:
            return parsed
    except Exception:
        pass

    try:
        return json.loads(response.output_text)
    except Exception:
        # Last resort
        return {"canonical": group_variants[0], "variants": group_variants}

def run_homogenization(unique_cols, api_key: str, embedding_model: str, llm_model: str, threshold: float):
    """
    Returns:
      canonical_map[col] -> canonical_name
      variants_map[col]  -> comma-separated variants in its group
    """
    texts = list(unique_cols)
    if not texts:
        return {}, {}

    vectors = _embed_texts(texts, api_key=api_key, embedding_model=embedding_model)
    sim = _cosine_sim_matrix(vectors)

    groups_idx = _union_find_groups(sim, threshold=threshold)

    canonical_map = {}
    variants_map = {}

    for idxs in groups_idx:
        group = [texts[i] for i in idxs]
        result = _pick_canonical_for_group(group_variants=group, api_key=api_key, llm_model=llm_model)

        canonical = str(result.get("canonical", "")).strip() or group[0]
        variants = result.get("variants", group)
        if not isinstance(variants, list) or len(variants) == 0:
            variants = group

        variants_str = ", ".join([str(v) for v in variants])

        for v in group:
            canonical_map[v] = canonical
            variants_map[v] = variants_str

    return canonical_map, variants_map


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Insurance Data Discovery", layout="wide")
st.title("Insurance Data Discovery Tool (Module 1)")
st.caption("Upload sample reports (Excel/CSV). We will extract column metadata and build a field inventory.")

# -------------------------------
# Inputs
# -------------------------------
source_system = st.text_input("Source System Name (e.g., Legacy PAS, Mainframe Claims)")

uploaded_files = st.file_uploader(
    "Upload report files",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

st.markdown("---")
st.subheader("Homogenization (AI)")

enable_homog = st.checkbox(
    "Enable AI Homogenization (group similar columns + recommend business-friendly canonical name)",
    value=True
)

# Prefer secrets/env over UI, but allow UI
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

# -------------------------------
# Main Processing
# -------------------------------
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

    profile_df = pd.DataFrame(profile_rows)
    st.dataframe(profile_df, use_container_width=True)

    # -------------------------------
    # Build Field-Level Inventory (Raw)
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
                    report_label = f"{f.name} | {sheet}"
                    for col in df.columns:
                        field_rows.append({"report_name": report_label, "column_original": str(col)})
        except Exception as e:
            st.error(f"Could not process {f.name}: {e}")

    field_df = pd.DataFrame(field_rows)

    # -------------------------------
    # AI Homogenization (optional): build maps
    # -------------------------------
    canonical_map, variants_map = {}, {}
    if enable_homog and not field_df.empty:
        st.write("## Homogenization (AI) Results (Collapsed to Canonical)")

        unique_cols = sorted(field_df["column_original"].dropna().astype(str).unique().tolist())

        with st.spinner("Running AI homogenization (embeddings + grouping + canonical naming)..."):
            try:
                canonical_map, variants_map = run_homogenization(
                    unique_cols=unique_cols,
                    api_key=api_key,
                    embedding_model=embedding_model.strip(),
                    llm_model=llm_model.strip(),
                    threshold=float(similarity_threshold),
                )
            except Exception as e:
                st.error(f"Homogenization failed: {e}")
                canonical_map, variants_map = {}, {}

    # -------------------------------
    # Cross Tab (Canonical in column_original + legacy_columns) + Totals Row + Repetition Count
    # REQUIRED OUTPUT:
    # - Index = canonical name (shown as column_original)
    # - legacy_columns = all other variants (excluding canonical)
    # - X present per report if any variant exists in that report
    # -------------------------------
    if not field_df.empty:
        st.write("## Report vs Field Cross Tab (X = Present)")

        if enable_homog and canonical_map:
            tmp = field_df.copy()
            tmp["canonical"] = tmp["column_original"].map(canonical_map).fillna(tmp["column_original"])

            # canonical -> variants set
            canonical_to_variants = {}
            for orig in tmp["column_original"].dropna().astype(str).unique():
                canon = canonical_map.get(orig, orig)
                v_str = variants_map.get(orig, orig)
                parts = [p.strip() for p in str(v_str).split(",") if p.strip()]
                canonical_to_variants.setdefault(canon, set()).update(parts)

            # canonical -> legacy (exclude canonical itself)
            canonical_to_legacy = {}
            for canon, varset in canonical_to_variants.items():
                canon_norm = str(canon).strip().lower()
                legacy_only = sorted([v for v in varset if str(v).strip().lower() != canon_norm])
                canonical_to_legacy[canon] = ", ".join(legacy_only)

            # collapse presence
            collapsed = (
                tmp.groupby(["report_name", "canonical"], as_index=False)
                   .size()
                   .drop(columns=["size"])
            )

            cross_counts = pd.crosstab(collapsed["canonical"], collapsed["report_name"])
            cross_tab = cross_counts.applymap(lambda v: "x" if v > 0 else "")

            # Put legacy next to canonical
            cross_tab.insert(
                loc=0,
                column="legacy_columns",
                value=cross_tab.index.to_series().map(canonical_to_legacy).fillna("")
            )

            # repetition across report columns only
            report_cols = [c for c in cross_tab.columns if c not in ["legacy_columns", "Repetition Count"]]
            cross_tab["Repetition Count"] = (cross_tab[report_cols] == "x").sum(axis=1)

            # totals row
            totals = (cross_tab[report_cols] == "x").sum(axis=0)
            totals["legacy_columns"] = ""
            totals["Repetition Count"] = int(cross_tab["Repetition Count"].sum())
            cross_tab.loc["Totals"] = totals

            cross_tab.index.name = "column_original"
            st.dataframe(cross_tab, use_container_width=True)

        else:
            # Non-AI fallback: raw columns
            cross_counts = pd.crosstab(field_df["column_original"], field_df["report_name"])
            cross_tab = cross_counts.applymap(lambda v: "x" if v > 0 else "")

            cross_tab["Repetition Count"] = (cross_tab == "x").sum(axis=1)

            totals = (cross_tab == "x").sum(axis=0)
            totals["Repetition Count"] = int(cross_tab["Repetition Count"].sum())
            cross_tab.loc["Totals"] = totals

            cross_tab.index.name = "column_original"
            st.dataframe(cross_tab, use_container_width=True)
