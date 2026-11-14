import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
from prepare_owt_expansion_v23_r2 import split_spans,acceptable_new_span

def test_abbreviation_boundary_preserves_U_S_and_named_titles():
    text='The diplomats in Iraq said the U.S. delegation was arriving on Friday. Dr. Smith discussed these results with the entire committee.'
    spans=list(split_spans(text))
    assert spans==['The diplomats in Iraq said the U.S. delegation was arriving on Friday.','Dr. Smith discussed these results with the entire committee.']
    assert all(acceptable_new_span(s) for s in spans)

def test_new_selection_excludes_obvious_fragment_and_dangling_abbreviation_only():
    assert not acceptable_new_span('forces and diplomats in Iraq, reaffirming that U.S.')
    assert not acceptable_new_span('The diplomats were waiting for the representative of the U.S.')
    assert acceptable_new_span('The diplomats were waiting for the United States delegation.')
    assert list(split_spans('A. Smith said the whole group would arrive soon. Another group arrived later that evening.'))==['A. Smith said the whole group would arrive soon.','Another group arrived later that evening.']

def test_normalized_spans_are_literal_source_substrings_and_numeric_date_is_not_abbreviation():
    text='They finished the road construction in December 2012. The local community celebrated the successful opening together.'
    assert len(list(split_spans(text)))==2
    assert all(s in ' '.join(text.split()) for s in split_spans(text))
    assert acceptable_new_span('“The local community celebrated the successful opening together.')
    assert not acceptable_new_span('This span contains replacement \ufffd and must be rejected.')


def test_month_abbreviation_is_not_a_sentence_end():
    text='Johnson and her daughter walked Yuri on a leash on Nov. 4 near the local park. The dog was very happy to be outside on this sunny afternoon.'
    spans=list(split_spans(text))
    assert spans==['Johnson and her daughter walked Yuri on a leash on Nov. 4 near the local park.','The dog was very happy to be outside on this sunny afternoon.']
    assert all(acceptable_new_span(x) for x in spans)
    assert not acceptable_new_span('Johnson and her daughter walked Yuri on a leash on Nov.')
