import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')


class EnhancedRadiologyDataPreprocessor:
    def __init__(self, data_path):
        """Initialize with path to the updated CSV file"""
        self.data_path = data_path
        self.df = None
        self.processed_df = None

        # Modality grouping mapping
        self.modality_mapping = {
            'MR': 'MR', 'CT': 'CT', 'CR': 'CR', 'US': 'US', 'XR': 'XR', 'NM': 'NM',
            'PET': 'OTHER', 'PT': 'OTHER', 'IR': 'OTHER', 'MG': 'OTHER',
            'FL': 'OTHER', 'DF': 'OTHER', 'DX': 'OTHER', 'MRA': 'OTHER'
        }

        # Findings-focused quality filtering criteria
        self.quality_filters = {
            'min_findings_length': 100,     # Focus on substantial findings only
            'min_impression_length': 20,
            'max_findings_length': 3000,
            'max_impression_length': 1000,
            'require_substantial_findings': True  # NEW: Must have meaningful findings
        }

    def convert_to_json(self, structured_report):
        """Convert structured report string to dictionary"""
        try:
            import ast
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

    def clean_text_field(self, text):
        """Clean individual text field"""
        if text is None:
            return None
        try:
            if pd.isna(text):
                return None
        except (ValueError, TypeError):
            pass

        text_str = str(text).strip()
        if text_str.lower() in ['nan', 'none', '']:
            return None

        # Remove electronic signature information
        signature_patterns = [
            r'Electronically signed by.*',
            r'Electronically Signed By.*',
            r'ELECTRONICALLY SIGNED BY.*',
            r'Interpreted by.*',
            r'INTERPRETED BY.*'
        ]

        for pattern in signature_patterns:
            text_str = re.sub(pattern, '', text_str, flags=re.IGNORECASE)

        text_str = re.sub(r'\s+', ' ', text_str).strip()

        if len(text_str) < 3 or text_str.lower() in ['findings:', 'impression:', 'history:', 'technique:']:
            return None

        return text_str

    def load_data(self):
        """Load the processed CSV file"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        print(f"Loaded {len(self.df)} reports")

        # Create mapped_clinic_id if it doesn't exist
        if 'mapped_clinic_id' not in self.df.columns and 'clinic_id' in self.df.columns:
            print("Creating mapped_clinic_id from clinic_id...")
            clinic_ids = self.df['clinic_id'].value_counts().index.tolist()
            mapped_clinic_ids = {
                clinic_id: f"clinic_{idx + 1}" for idx, clinic_id in enumerate(clinic_ids)}
            self.df['mapped_clinic_id'] = self.df['clinic_id'].map(
                mapped_clinic_ids)

        return self

    def group_modalities(self):
        """Group modalities into major/minor/other categories"""
        print("Grouping modalities...")
        self.df['grouped_modality'] = self.df['modality'].map(
            self.modality_mapping)
        modality_dist = self.df['grouped_modality'].value_counts()
        print("Modality distribution after grouping:")
        print(modality_dist)
        return self

    def extract_essential_clinical_fields(self):
        """Extract only essential clinical information: findings and impression"""
        print("Extracting essential clinical fields (findings and impression only)...")

        # Initialize only essential columns
        essential_fields = ['findings', 'impression']

        for field in essential_fields:
            self.df[f'clean_{field}'] = None

        # Extract from structured_report_dict
        if 'structured_report_dict' in self.df.columns:
            print("Extracting from structured_report_dict...")
            for idx, row in self.df.iterrows():
                if idx % 5000 == 0:
                    print(f"Processed {idx} reports...")

                report_dict = self.convert_to_json(
                    row['structured_report_dict'])

                # Extract findings and impression only
                for field in essential_fields:
                    content = report_dict.get(field, '')
                    if isinstance(content, list):
                        content = ' '.join([str(f)
                                           for f in content if str(f).strip()])
                    if content:
                        cleaned = self.clean_text_field(content)
                        if cleaned:
                            self.df.at[idx, f'clean_{field}'] = cleaned

        return self

    def create_findings_focused_content(self):
        """Create content focused only on substantial findings"""
        print("Creating findings-focused content...")

        self.df['composite_findings'] = None
        self.df['findings_length'] = 0
        self.df['total_clinical_length'] = 0

        for idx, row in self.df.iterrows():
            # Only use findings - no metadata supplementation
            if pd.notna(row['clean_findings']) and len(str(row['clean_findings']).strip()) >= self.quality_filters['min_findings_length']:
                findings_text = f"FINDINGS: {row['clean_findings']}"
                self.df.at[idx, 'composite_findings'] = findings_text
                self.df.at[idx, 'findings_length'] = len(row['clean_findings'])
                self.df.at[idx, 'total_clinical_length'] = len(
                    row['clean_findings'])
            else:
                # Mark as insufficient findings
                self.df.at[idx, 'composite_findings'] = None
                self.df.at[idx, 'findings_length'] = 0
                self.df.at[idx, 'total_clinical_length'] = 0

        return self

    def apply_enhanced_quality_filters(self):
        """Apply enhanced quality filtering"""
        print("Applying enhanced quality filters...")

        initial_count = len(self.df)

        # Basic required fields
        mask = (
            self.df['clean_impression'].notna() &
            self.df['composite_findings'].notna()
        )

        self.df = self.df[mask].copy()
        print(
            f"After basic filtering: {len(self.df)} records ({initial_count - len(self.df)} removed)")

        # Enhanced length filters
        length_mask = (
            (self.df['clean_impression'].str.len() >= self.quality_filters['min_impression_length']) &
            (self.df['clean_impression'].str.len() <= self.quality_filters['max_impression_length']) &
            (self.df['findings_length'] >=
             self.quality_filters['min_findings_length'])
        )

        filtered_count = len(self.df)
        self.df = self.df[length_mask].copy()
        print(
            f"After enhanced length filtering: {len(self.df)} records ({filtered_count - len(self.df)} removed)")
        print(
            f"Total retained: {len(self.df)/initial_count*100:.1f}% of original data")

        return self

    def create_enhanced_input(self, row):
        """Create enhanced input with findings-focused content"""
        context_parts = []

        # Always include clinic and modality
        context_parts.append(f"[CLINIC: {row['mapped_clinic_id']}]")
        context_parts.append(f"[MODALITY: {row['grouped_modality']}]")

        # Add only the findings content
        context_parts.append(row['composite_findings'])
        context_parts.append("IMPRESSION:")

        return " ".join(context_parts)

    def format_for_training(self):
        """Format data for fine-tuning"""
        print("Formatting data for training...")

        # Create training input and output
        self.df['training_input'] = self.df.apply(
            self.create_enhanced_input, axis=1)
        self.df['training_output'] = self.df['clean_impression']

        # Create clinic-modality combination identifier
        self.df['clinic_modality'] = (
            self.df['mapped_clinic_id'].astype(str) + "_" +
            self.df['grouped_modality'].astype(str)
        )

        return self

    def analyze_segments(self):
        """Analyze clinic-modality segments for stratified splitting"""
        print("\nAnalyzing clinic-modality segments:")

        segment_analysis = self.df.groupby('clinic_modality').agg({
            'report_id': 'count',
            'mapped_clinic_id': 'first',
            'grouped_modality': 'first'
        }).rename(columns={'report_id': 'count'})

        segment_analysis = segment_analysis.sort_values(
            'count', ascending=False)
        print(segment_analysis)

        # Categorize segments by size
        large_segments = segment_analysis[segment_analysis['count'] >= 100].index
        medium_segments = segment_analysis[
            (segment_analysis['count'] >= 20) & (
                segment_analysis['count'] < 100)
        ].index
        small_segments = segment_analysis[segment_analysis['count'] < 20].index

        print(f"\nSegment categories:")
        print(f"Large segments (≥100): {len(large_segments)}")
        print(f"Medium segments (20-99): {len(medium_segments)}")
        print(f"Small segments (<20): {len(small_segments)}")

        return segment_analysis

    def stratified_split(self, test_size=0.15, val_size=0.15, random_state=42):
        """Create stratified train/validation/test splits"""
        print("Creating stratified splits...")

        train_data = []
        val_data = []
        test_data = []

        for clinic_modality in self.df['clinic_modality'].unique():
            segment_data = self.df[self.df['clinic_modality']
                                   == clinic_modality].copy()
            segment_size = len(segment_data)

            if segment_size < 5:
                if segment_size >= 3:
                    train_data.append(segment_data.iloc[:-2])
                    val_data.append(segment_data.iloc[-2:-1])
                    test_data.append(segment_data.iloc[-1:])
                else:
                    train_data.append(segment_data)

            elif segment_size < 20:
                # Small segments: 60/20/20 split
                train_seg, temp = train_test_split(
                    segment_data, test_size=0.4, random_state=random_state, shuffle=True
                )
                if len(temp) >= 2:
                    val_seg, test_seg = train_test_split(
                        temp, test_size=0.5, random_state=random_state, shuffle=True
                    )
                else:
                    val_seg = temp.iloc[:1] if len(
                        temp) > 0 else pd.DataFrame()
                    test_seg = temp.iloc[1:] if len(
                        temp) > 1 else pd.DataFrame()

                train_data.append(train_seg)
                val_data.append(val_seg)
                test_data.append(test_seg)

            else:
                train_seg, temp = train_test_split(
                    segment_data,
                    test_size=(test_size + val_size),
                    random_state=random_state,
                    shuffle=True
                )
                val_seg, test_seg = train_test_split(
                    temp,
                    test_size=(test_size / (test_size + val_size)),
                    random_state=random_state,
                    shuffle=True
                )

                train_data.append(train_seg)
                val_data.append(val_seg)
                test_data.append(test_seg)

        # Combine all segments
        train_df = pd.concat(
            train_data, ignore_index=True) if train_data else pd.DataFrame()
        val_df = pd.concat(
            val_data, ignore_index=True) if val_data else pd.DataFrame()
        test_df = pd.concat(
            test_data, ignore_index=True) if test_data else pd.DataFrame()

        print(f"Final split sizes:")
        print(
            f"Train: {len(train_df)} ({len(train_df)/len(self.df)*100:.1f}%)")
        print(
            f"Validation: {len(val_df)} ({len(val_df)/len(self.df)*100:.1f}%)")
        print(f"Test: {len(test_df)} ({len(test_df)/len(self.df)*100:.1f}%)")

        return train_df, val_df, test_df

    def create_reference_banks(self, train_df):
        """Create reference text banks for style learning"""
        print("Creating reference text banks...")

        reference_banks = {}

        for clinic_id in train_df['mapped_clinic_id'].unique():
            clinic_impressions = train_df[
                train_df['mapped_clinic_id'] == clinic_id
            ]['clean_impression'].tolist()

            # Sample a subset for efficiency (max 100 per clinic)
            if len(clinic_impressions) > 100:
                clinic_impressions = np.random.choice(
                    clinic_impressions, 100, replace=False
                ).tolist()

            reference_banks[clinic_id] = clinic_impressions

        print(f"Created reference banks for {len(reference_banks)} clinics")
        return reference_banks

    def save_processed_data(self, train_df, val_df, test_df, reference_banks, output_dir="./enhanced_processed_data"):
        """Save all processed data"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving enhanced processed data to {output_dir}...")

        # Save splits
        train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
        val_df.to_csv(f"{output_dir}/val_data.csv", index=False)
        test_df.to_csv(f"{output_dir}/test_data.csv", index=False)

        # Save reference banks
        with open(f"{output_dir}/reference_banks.json", 'w') as f:
            json.dump(reference_banks, f, indent=2)

        # Save training examples in JSONL format for OpenAI fine-tuning
        def save_jsonl(df, filename):
            with open(f"{output_dir}/{filename}", 'w') as f:
                for _, row in df.iterrows():
                    example = {
                        "messages": [
                            {"role": "user", "content": row['training_input']},
                            {"role": "assistant",
                                "content": row['training_output']}
                        ]
                    }
                    f.write(json.dumps(example) + '\n')

        save_jsonl(train_df, "train_data.jsonl")
        save_jsonl(val_df, "val_data.jsonl")

        print("Enhanced data processing complete!")

        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(
            f"Total processed records: {len(train_df) + len(val_df) + len(test_df)}")
        print(
            f"Clinic-modality combinations: {train_df['clinic_modality'].nunique()}")
        print(f"Clinics: {train_df['mapped_clinic_id'].nunique()}")
        print(f"Modalities: {train_df['grouped_modality'].nunique()}")

        # Show examples of enhanced inputs
        print(f"\n=== SAMPLE ENHANCED INPUTS ===")
        for i, (_, row) in enumerate(train_df.sample(3).iterrows()):
            print(f"\nSample {i+1}:")
            print(f"Input: {row['training_input'][:200]}...")
            print(f"Output: {row['training_output'][:100]}...")

    def run_enhanced_pipeline(self):
        """Run the complete findings-focused preprocessing pipeline"""
        # Load and process data
        self.load_data()
        self.group_modalities()
        self.extract_essential_clinical_fields()
        self.create_findings_focused_content()
        self.apply_enhanced_quality_filters()
        self.format_for_training()

        # # Analyze segments
        # segment_analysis = self.analyze_segments()

        # Create splits
        train_df, val_df, test_df = self.stratified_split()

        # Create reference banks
        reference_banks = self.create_reference_banks(train_df)

        # Save everything
        self.save_processed_data(train_df, val_df, test_df, reference_banks)

        return train_df, val_df, test_df, reference_banks


# Usage example
if __name__ == "__main__":
    # Initialize enhanced preprocessor
    preprocessor = EnhancedRadiologyDataPreprocessor(
        "updated_deid_reports.csv")

    # Run enhanced pipeline
    train_df, val_df, test_df, reference_banks = preprocessor.run_enhanced_pipeline()
