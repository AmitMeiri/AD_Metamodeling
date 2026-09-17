import pandas as pd

df4 = pd.read_csv('real_data_output/real_data_classification_4subtypes.csv')
df6 = pd.read_csv('real_data_output/real_data_classification_6subtypes.csv')

# Merge on PTID and VISCODE2
merged = pd.merge(
    df4[['PTID', 'VISCODE2', 'Subtype_Number', 'Subtype_Label', 'Stage_Number']],
    df6[['PTID', 'VISCODE2', 'Subtype_Number', 'Subtype_Label', 'Stage_Number']],
    on=['PTID', 'VISCODE2'],
    suffixes=('_k4', '_k6')
)

# Cross-tabulation of positive tau scans (Stage > 0)
tau_pos = merged[(merged['Subtype_Number_k4'] > 0) & (merged['Subtype_Number_k6'] > 0)]

print("Cross-tabulation between k=4 Subtypes and k=6 Subtypes (Scan Counts):")
ct = pd.crosstab(
    merged['Subtype_Label_k4'],
    merged['Subtype_Label_k6'],
    margins=True
)
print(ct.to_string())

print("\n" + "="*80)
print("Lineage breakdown for each k=4 Subtype into k=6 Subtypes:")
for k4_val in sorted(merged['Subtype_Number_k4'].unique()):
    sub_df = merged[merged['Subtype_Number_k4'] == k4_val]
    k4_name = sub_df['Subtype_Label_k4'].iloc[0]
    print(f"\nParent in k=4: {k4_name} (Total: {len(sub_df)} scans)")
    counts = sub_df['Subtype_Label_k6'].value_counts()
    for label, cnt in counts.items():
        pct = (cnt / len(sub_df)) * 100
        print(f"  --> {label}: {cnt} scans ({pct:.1f}%)")
