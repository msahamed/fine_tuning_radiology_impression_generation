import pandas as pd
import ast

def convert_to_json(structured_report):
    try:
        if pd.isna(structured_report) or not isinstance(structured_report, str):
            return {}
        parsed_dict = ast.literal_eval(structured_report)
        clean_dict = {}
        for key_tuple, value_list in parsed_dict.items():
            if not isinstance(key_tuple, tuple) or len(key_tuple) != 1:
                continue
            key = key_tuple[0].rstrip(':').lower().strip()
            value = value_list[0] if len(value_list) == 1 else value_list
            clean_dict[key] = value
        return clean_dict
    except (SyntaxError, ValueError, TypeError):
        return {}

# Load original data
df = pd.read_csv("updated_deid_reports.csv")
print(f"Original dataset: {len(df)} reports")

# Create mapped_clinic_id
if 'mapped_clinic_id' not in df.columns and 'clinic_id' in df.columns:
    clinic_ids = df['clinic_id'].value_counts().index.tolist()
    mapped_clinic_ids = {clinic_id: f"clinic_{idx + 1}" for idx, clinic_id in enumerate(clinic_ids)}
    df['mapped_clinic_id'] = df['clinic_id'].map(mapped_clinic_ids)

# Analyze each filtering step
print("\n=== FILTERING ANALYSIS ===")

# Check basic field availability
basic_fields_available = (
    df['mapped_clinic_id'].notna() &
    df['modality'].notna() &
    df['structured_report_dict'].notna()
)
print(f"Reports with basic fields (clinic, modality, structured_report): {basic_fields_available.sum()}")
print(f"Missing basic fields: {len(df) - basic_fields_available.sum()}")

# Extract structured fields for analysis
df_analysis = df[basic_fields_available].copy()
df_analysis['has_findings'] = False
df_analysis['has_impression'] = False

print("\nAnalyzing structured_report_dict extraction...")
for idx, row in df_analysis.iterrows():
    report_dict = convert_to_json(row['structured_report_dict'])

    # Check findings
    findings = report_dict.get('findings', '')
    if findings:
        if isinstance(findings, list):
            findings = ' '.join([str(f) for f in findings if str(f).strip()])
        if findings and len(str(findings).strip()) >= 20:  # Our minimum length
            df_analysis.at[idx, 'has_findings'] = True

    # Check impression
    impression = report_dict.get('impression', '')
    if impression:
        if isinstance(impression, list):
            impression = ' '.join([str(i) for i in impression if str(i).strip()])
        if impression and len(str(impression).strip()) >= 10:  # Our minimum length
            df_analysis.at[idx, 'has_impression'] = True

print(f"\nStructured field extraction results:")
print(f"Reports with valid findings: {df_analysis['has_findings'].sum()}")
print(f"Reports with valid impression: {df_analysis['has_impression'].sum()}")
print(f"Reports with both findings AND impression: {(df_analysis['has_findings'] & df_analysis['has_impression']).sum()}")

# Final quality filter
final_valid = (
    df_analysis['has_findings'] &
    df_analysis['has_impression']
)

print(f"\n=== FINAL RESULTS ===")
print(f"Reports passing all filters: {final_valid.sum()}")
print(f"Reports filtered out: {len(df) - final_valid.sum()}")

# Breakdown of what was filtered
print(f"\n=== BREAKDOWN OF FILTERED REPORTS ===")
print(f"Missing basic fields (clinic/modality/structured_report): {len(df) - basic_fields_available.sum()}")
print(f"Missing or invalid findings: {basic_fields_available.sum() - df_analysis['has_findings'].sum()}")
print(f"Missing or invalid impression: {basic_fields_available.sum() - df_analysis['has_impression'].sum()}")
print(f"Missing both findings and impression: {basic_fields_available.sum() - (df_analysis['has_findings'] | df_analysis['has_impression']).sum()}")

# Sample of what was filtered out
print(f"\n=== SAMPLE OF FILTERED REPORTS ===")
filtered_out = df_analysis[~final_valid].sample(min(3, len(df_analysis[~final_valid])))
for idx, row in filtered_out.iterrows():
    report_dict = convert_to_json(row['structured_report_dict'])
    print(f"\nReport {idx}:")
    print(f"  Clinic: {row['mapped_clinic_id']}")
    print(f"  Modality: {row['modality']}")
    print(f"  Has findings: {row['has_findings']}")
    print(f"  Has impression: {row['has_impression']}")
    print(f"  Available keys: {list(report_dict.keys())[:5]}...")  # First 5 keys