#!/usr/bin/env python3
"""
Test script for the option filter helper functions
"""

from main import get_filtered_options_by_date, format_options_for_chat, clear_cache

def test_option_filter():
    """Test the option filter helper function"""

    print("Testing option filter helper functions...")
    print("=" * 60)

    # Clear cache to ensure fresh data
    clear_cache()

    # Example 1: Get UAMY calls in date range 11/25-1/26
    print("\nExample 1: UAMY calls from 11/25 to 1/26")
    print("-" * 60)

    results = get_filtered_options_by_date(
        ticker="UAMY",
        date_range="11/25-1/26",
        min_volume=10,  # Lower thresholds for testing
        min_oi=50,
        option_type="calls",
        include_greeks=True
    )

    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        formatted = format_options_for_chat(results, include_greeks=True)
        print(formatted)

    print("\n" + "=" * 60)
    print("\nExample 2: UAMY calls from 12/1 to 12/31 (without Greeks)")
    print("-" * 60)

    results2 = get_filtered_options_by_date(
        ticker="UAMY",
        date_range="12/1-12/31",
        min_volume=10,
        min_oi=50,
        option_type="calls",
        include_greeks=False
    )

    if "error" in results2:
        print(f"Error: {results2['error']}")
    else:
        formatted2 = format_options_for_chat(results2, include_greeks=False)
        print(formatted2)

if __name__ == "__main__":
    test_option_filter()
