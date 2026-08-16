from train import encode_ordinal, load_data


def test_mapping_matches_contract():
    """Stichproben aus der Encoding-Tabelle, inklusive der alphabetischen Falle."""
    _, encoding = encode_ordinal(load_data())

    assert encoding["purpose"]["A410"] == 2
    assert encoding["purpose"]["A48"] == 8
    assert encoding["purpose"]["A49"] == 9
    assert encoding["housing"]["A152"] == 1
    assert encoding["housing"]["A153"] == 2


def test_input_is_not_modified():
    """encode_ordinal arbeitet auf einer Kopie."""
    df = load_data()
    before = df["housing"].iloc[0]

    encode_ordinal(df)

    assert df["housing"].iloc[0] == before


def test_no_text_columns_remain():
    encoded, _ = encode_ordinal(load_data())

    assert not any(encoded[col].dtype == "object" for col in encoded.columns)