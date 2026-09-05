from datetime import date

import pytest

from optionspricing.market_data import ParsedOccTicker, parse_occ_ticker


def test_parses_a_standard_call_ticker():
    assert parse_occ_ticker("AAPL241220C00255000") == ParsedOccTicker(
        root="AAPL", expiry=date(2024, 12, 20), option_type="call", strike=255.0
    )


def test_parses_a_put_ticker():
    assert parse_occ_ticker("GOOGL250117P00150000") == ParsedOccTicker(
        root="GOOGL", expiry=date(2025, 1, 17), option_type="put", strike=150.0
    )


def test_root_symbol_is_not_fixed_width():
    # The point of parsing with a regex rather than a fixed 4-character slice:
    # roots are 1-5+ characters, and weekly/index roots like SPXW aren't 4 either.
    roots = ["F", "IBM", "AAPL", "GOOGL", "SPXW"]

    for root in roots:
        parsed = parse_occ_ticker(f"{root}241220C00255000")
        assert parsed.root == root
        assert parsed.strike == 255.0


def test_strike_is_scaled_by_one_thousand():
    # The 8-digit strike field is strike * 1000, so it carries fractional strikes.
    assert parse_occ_ticker("F241220C00007500").strike == 7.5
    assert parse_occ_ticker("SPXW241220C05000000").strike == 5000.0


def test_input_is_upper_cased_and_stripped():
    assert parse_occ_ticker("  aapl241220c00255000  ") == parse_occ_ticker("AAPL241220C00255000")


def test_rejects_malformed_tickers():
    malformed = [
        "",
        "AAPL",
        "AAPL241220C0025500",   # 7-digit strike
        "AAPL2412C00255000",    # 4-digit expiry
        "AAPL241220X00255000",  # neither call nor put
        "241220C00255000",      # no root symbol
        "AAPL241220C00255000X",  # trailing junk
    ]

    for ticker in malformed:
        with pytest.raises(ValueError):
            parse_occ_ticker(ticker)
