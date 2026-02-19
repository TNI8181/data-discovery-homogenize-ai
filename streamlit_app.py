import streamlit as st
import pandas as pd
import io
import numpy as np
import json

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
    # Normalize for cosine similarity
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

    # Union edges above threshold (upper triangle)
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
    Cached by (texts, embedding_model). api_key is included only to satisfy Streamlit cache signature.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing dependency: openai. Install with `pip install openai`") from e

    client = OpenAI(api_key=api_key)
    # One batch call
    resp = client.embeddings.create(
        model=embedding_model,
        input=texts
    )
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
            "variants": {
                "type": "array",
                "items": {"type": "string"}
            }
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
                "name": "homogenization_result",   # ✅ REQUIRED
                "type": "json_schema",
                "schema": schema                  # ✅ schema lives here in this variant
            }
        }
    )

    # This should now be parsed JSON (dict) if the SDK supports output_parsed
    try:
        return response.output_parsed
    except Exception:
        # Fallback: parse raw text
        import json
        return json.loads(response.output_text)

    prompt = f"""
You are helping homogenize legacy data field names for an insurance data discovery tool.

Given a list of column name variants that likely mean the same business concept, do BOTH:
1) Choose the best canonical name for business users: clear, readable, Title Case, with spaces (NOT snake_case).
2) Return the variants exactly as provided (same spelling/case), in any order.

Variants:
{group_variants}

Rules for the canonical name:
- Prefer the most readable business label (Title Case with spaces), e.g., "Policy Number"
- Avoid abbreviations if a clear expanded form exists
- Avoid underscores and camelCase in the canonical name
"""

    resp = client.responses.create(
        model=llm_model,
        input=[
            {"role": "system", "content": "Return ONLY valid JSON that matches the schema."},
            {"role": "user", "content": prompt}
        ],
        text={"format": schema}
    )

    # SDK convenience property: output_text is the JSON string
    data = json.loads(resp.output_text)
    return data

def run_homogenization(unique_cols, api_key: str, embedding_model: str, llm_model: str, threshold: float):
    """
    Returns:
      canonical_map[col] -> canonical_name
      variants_map[col]  -> comma-separated variants in its group
    """
    texts = list(unique_cols)
    if len(texts) == 0:
        return {}, {}

    # Embeddings
    vectors = _embed_texts(texts, api_key=api_key, embedding_model=embedding_model)
    sim = _cosine_sim_matrix(vectors)

    # Grouping
    groups_idx = _union_find_groups(sim, threshold=threshold)

    canonical_map = {}
    variants_map = {}

    # LLM canonical selection per group
    for idxs in groups_idx:
        group = [texts[i] for i in idxs]
        # Small groups of 1: canonical is just a cleaned Title Case guess via LLM anyway (keeps approach consistent)
        result = _pick_canonical_for_group(group_variants=group, api_key=api_key, llm_model=llm_model)
        canonical = result.get("canonical", "").strip() or group[0]
        variants = result.get("variants", group)

        variants_str = ", ".join(variants)

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

enable_homog = st.checkbox("Enable AI Homogenization (group similar columns + recommend business-friendly canonical name)", value=True)

api_key = st.text_input(
    "OpenAI API Key (recommended: set as environment variable in production)",
    type="password"
)

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

    if enable_homog and not api_key.strip():
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
    # Build Field-Level Inventory (Raw Only)
    # -------------------------------
    st.write("## Field Inventory (Raw)")
    field_rows = []

    for f in uploaded_files:
        try:
            if f.name.lower().endswith(".csv"):
                df = read_csv_flexible(f)
                report_label = f.name
                for col in df.columns:
                    field_rows.append({
                        "report_name": report_label,
                        "column_original": str(col)
                    })
            else:
                xls = pd.ExcelFile(f)
                for sheet in xls.sheet_names:
                    df = xls.parse(sheet)
                    report_label = f"{f.name} | {sheet}"
                    for col in df.columns:
                        field_rows.append({
                            "report_name": report_label,
                            "column_original": str(col)
                        })
        except Exception as e:
            st.error(f"Could not process {f.name}: {e}")

    field_df = pd.DataFrame(field_rows)

    # -------------------------------
    # AI Homogenization: add 2 columns next to column_original
    #   - recommended_canonical (Option 1: business-readable)
    #   - similar_variants (comma-separated)
    # -------------------------------
    if enable_homog and not field_df.empty:
        st.write("## Homogenization (AI) Results")

        unique_cols = sorted(field_df["column_original"].dropna().astype(str).unique().tolist())

        with st.spinner("Running AI homogenization (embeddings + grouping + canonical naming)..."):
            try:
                canonical_map, variants_map = run_homogenization(
                    unique_cols=unique_cols,
                    api_key=api_key.strip(),
                    embedding_model=embedding_model.strip(),
                    llm_model=llm_model.strip(),
                    threshold=float(similarity_threshold),
                )
            except Exception as e:
                st.error(f"Homogenization failed: {e}")
                canonical_map, variants_map = {}, {}

        field_df["recommended_canonical"] = field_df["column_original"].map(canonical_map).fillna("")
        field_df["similar_variants"] = field_df["column_original"].map(variants_map).fillna("")

        # Put these right after column_original
        desired_cols = ["report_name", "column_original", "recommended_canonical", "similar_variants"]
        field_df = field_df[[c for c in desired_cols if c in field_df.columns] +
                            [c for c in field_df.columns if c not in desired_cols]]

    st.dataframe(field_df, use_container_width=True)

   # -------------------------------
# Cross Tab (Canonical in column_original + legacy_columns) + Totals Row + Repetition Count
# Logic:
# - AI gives canonical_map (orig -> canonical business label) and variants_map (orig -> "v1, v2, ...")
# - We collapse all variants into ONE row per canonical field on the cross tab
# - cross tab index shows ONLY the canonical name (column_original)
# - legacy_columns shows ONLY the other variants (excluding the canonical itself), comma-separated
# - X is present in a report if ANY variant for that canonical appears in that report
# -------------------------------
if not field_df.empty:
    st.write("## Report vs Field Cross Tab (X = Present)")

    # If homogenization is enabled and we have AI results, collapse onto canonical
    if enable_homog:
        # 1) Map each original column to its canonical business label
        tmp = field_df.copy()
        tmp["canonical"] = tmp["column_original"].map(canonical_map).fillna(tmp["column_original"])

        # 2) Build canonical -> set(all variants) using variants_map (comma-separated strings)
        canonical_to_variants = {}
        for orig in tmp["column_original"].dropna().astype(str).unique():
            canon = canonical_map.get(orig, orig)
            v_str = variants_map.get(orig, orig)
            parts = [p.strip() for p in str(v_str).split(",") if p.strip()]
            canonical_to_variants.setdefault(canon, set()).update(parts)

        # 3) canonical -> legacy_columns (exclude canonical itself, case-insensitive)
        canonical_to_legacy = {}
        for canon, varset in canonical_to_variants.items():
            canon_norm = str(canon).strip().lower()
            legacy_only = sorted([v for v in varset if str(v).strip().lower() != canon_norm])
            canonical_to_legacy[canon] = ", ".join(legacy_only)

        # 4) Collapse presence: one row per (report_name, canonical)
        collapsed = (
            tmp.groupby(["report_name", "canonical"], as_index=False)
               .size()
               .drop(columns=["size"])
        )

        # 5) Crosstab on canonical (this becomes column_original)
        cross_counts = pd.crosstab(
            collapsed["canonical"],
            collapsed["report_name"]
        )

        cross_tab = cross_counts.applymap(lambda v: "x" if v > 0 else "")

        # 6) Insert legacy_columns next to canonical index
        cross_tab.insert(
            loc=0,
            column="legacy_columns",
            value=cross_tab.index.to_series().map(canonical_to_legacy).fillna("")
        )

        # 7) Repetition Count (count of x across report columns only)
        report_cols = [c for c in cross_tab.columns if c not in ["legacy_columns", "Repetition Count"]]
        cross_tab["Repetition Count"] = (cross_tab[report_cols] == "x").sum(axis=1)

        # 8) Totals row (count of x per report column; blank legacy; repetition total)
        totals = (cross_tab[report_cols] == "x").sum(axis=0)
        totals["legacy_columns"] = ""
        totals["Repetition Count"] = int(cross_tab["Repetition Count"].sum())
        cross_tab.loc["Totals"] = totals

        # Rename index label visually to match your requirement
        cross_tab.index.name = "column_original"

        st.dataframe(cross_tab, use_container_width=True)

    else:
        # No homogenization: use raw column_original as-is
        cross_counts = pd.crosstab(
            field_df["column_original"],
            field_df["report_name"]
        )
        cross_tab = cross_counts.applymap(lambda v: "x" if v > 0 else "")

        cross_tab["Repetition Count"] = (cross_tab == "x").sum(axis=1)

        totals = (cross_tab == "x").sum(axis=0)
        totals["Repetition Count"] = int(cross_tab["Repetition Count"].sum())
        cross_tab.loc["Totals"] = totals

        cross_tab.index.name = "column_original"
        st.dataframe(cross_tab, use_container_width=True)
