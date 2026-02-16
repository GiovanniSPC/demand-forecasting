import pandas as pd


class DataValidator:
    """
    Validates the cleaned dataset against the raw dataset.

    Checks performed:
        1. Item count comparison
        2. Category count comparison
        3. Total quantity comparison
        4. Quantity per item comparison
        5. Quantity per category comparison
        6. Date continuity check (no gaps in cleaned data)

    Usage:
        validator = DataValidator(df_raw, df_clean)
        validator.validate()
    """

    def __init__(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame):
        """
        Args:
            df_raw:   Raw DataFrame (before cleaning).
            df_clean: Cleaned DataFrame (after cleaning).
        """
        self.df_raw = df_raw.copy()
        self.df_clean = df_clean.copy()
        self.results = {}

    def _check_item_count(self) -> bool:
        """Compare the number of unique items between raw and clean."""
        raw_count = self.df_raw['item_number'].nunique()
        clean_count = self.df_clean['item_number'].nunique()
        passed = raw_count == clean_count

        self.results['item_count'] = {
            'raw': raw_count,
            'clean': clean_count,
            'passed': passed
        }
        return passed

    def _check_category_count(self) -> bool:
        """Compare the number of unique categories between raw and clean."""
        raw_count = self.df_raw['category'].nunique()
        clean_count = self.df_clean['category'].nunique()
        passed = raw_count == clean_count

        self.results['category_count'] = {
            'raw': raw_count,
            'clean': clean_count,
            'passed': passed
        }
        return passed

    def _check_total_quantity(self) -> bool:
        """Compare total quantity between raw and clean."""
        raw_total = self.df_raw['quantity'].sum()
        clean_total = self.df_clean['quantity'].sum()
        passed = raw_total == clean_total

        self.results['total_quantity'] = {
            'raw': raw_total,
            'clean': clean_total,
            'diff': clean_total - raw_total,
            'passed': passed
        }
        return passed

    def _check_quantity_per_item(self) -> pd.DataFrame:
        """Compare total quantity per item between raw and clean."""
        raw_qty = self.df_raw.groupby('item_number')['quantity'].sum().rename('raw_qty')
        clean_qty = self.df_clean.groupby('item_number')['quantity'].sum().rename('clean_qty')

        comp = pd.concat([raw_qty, clean_qty], axis=1).fillna(0)
        comp['diff'] = comp['clean_qty'] - comp['raw_qty']
        comp['match'] = comp['diff'] == 0

        mismatches = comp[~comp['match']]
        self.results['quantity_per_item'] = {
            'total_items': len(comp),
            'mismatches': len(mismatches),
            'passed': len(mismatches) == 0,
            'details': mismatches if len(mismatches) > 0 else None
        }
        return comp

    def _check_quantity_per_category(self) -> pd.DataFrame:
        """Compare total quantity per category between raw and clean."""
        raw_qty = self.df_raw.groupby('category')['quantity'].sum().rename('raw_qty')
        clean_qty = self.df_clean.groupby('category')['quantity'].sum().rename('clean_qty')

        comp = pd.concat([raw_qty, clean_qty], axis=1).fillna(0)
        comp['diff'] = comp['clean_qty'] - comp['raw_qty']
        comp['match'] = comp['diff'] == 0

        mismatches = comp[~comp['match']]
        self.results['quantity_per_category'] = {
            'total_categories': len(comp),
            'mismatches': len(mismatches),
            'passed': len(mismatches) == 0,
            'details': mismatches if len(mismatches) > 0 else None
        }
        return comp

    def _check_date_continuity(self) -> bool:
        """Verify there are no gaps in the cleaned dataset's date range."""
        dates = pd.to_datetime(self.df_clean['date'])
        date_range = pd.date_range(start=dates.min(), end=dates.max(), freq='D')
        actual_dates = dates.dt.normalize().unique()
        missing = set(date_range) - set(actual_dates)

        self.results['date_continuity'] = {
            'expected_days': len(date_range),
            'actual_days': len(actual_dates),
            'missing_days': len(missing),
            'passed': len(missing) == 0
        }
        return len(missing) == 0

    def _print_result(self, name: str, result: dict):
        """Pretty print a single check result."""
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"\n  {status} — {name}")
        for k, v in result.items():
            if k in ('passed', 'details'):
                continue
            print(f"      {k}: {v}")
        if not result['passed'] and result.get('details') is not None:
            print(f"      details:\n{result['details']}")

    def validate(self) -> bool:
        """
        Run all validation checks and print a summary report.

        Returns:
            True if all checks passed, False otherwise.
        """
        print("=" * 60)
        print("  DATA VALIDATION REPORT")
        print("=" * 60)

        self._check_item_count()
        self._check_category_count()
        self._check_total_quantity()
        self._check_quantity_per_item()
        self._check_quantity_per_category()
        self._check_date_continuity()

        for name, result in self.results.items():
            self._print_result(name, result)

        all_passed = all(r['passed'] for r in self.results.values())

        print("\n" + "=" * 60)
        if all_passed:
            print("  ✅ ALL CHECKS PASSED")
        else:
            failed = [k for k, v in self.results.items() if not v['passed']]
            print(f"  ❌ {len(failed)} CHECK(S) FAILED: {', '.join(failed)}")
        print("=" * 60)

        return all_passed


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from data_cleaner import DataCleaner

    # Clean the data
    cleaner = DataCleaner("/mnt/user-data/uploads/Transaction_Details.xlsx")
    df_clean = cleaner.clean()

    # Standardize raw columns to match
    df_raw = cleaner.df_raw.copy()
    df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_')

    # Validate
    validator = DataValidator(df_raw, df_clean)
    validator.validate()
