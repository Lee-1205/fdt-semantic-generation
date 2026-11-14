"""Semantic checks after tokenization/cropping, independent of tensor validity."""
import ast
import json
import re
import textwrap


def retrieval_contract(prompt,answer):
    queries=re.findall(r'Copy only the complete value stored at (K\d+)\.',prompt)
    if len(queries)!=1:return 'QUERY_NOT_UNIQUE'
    key=queries[0]
    values=re.findall(r'(?:^|\n)'+re.escape(key)+r'=([^\r\n]+)',prompt)
    if not values:return 'SOURCE_NOT_VISIBLE'
    if len(values)!=1:return 'SOURCE_NOT_UNIQUE'
    return 'PASS' if values[0].strip()==answer.strip() else 'VISIBLE_SOURCE_TARGET_MISMATCH'


def python_contract(answer):
    try:ast.parse(textwrap.dedent(answer).strip())
    except (SyntaxError,ValueError):return 'INVALID_PYTHON_SYNTAX'
    return 'PASS_SYNTAX_ONLY'


def json_contract(answer):
    try:json.loads(answer)
    except (ValueError,TypeError):return 'INVALID_JSON'
    return 'PASS_SYNTAX_ONLY'


def require_semantic_clearance(plan_dir):
    if (plan_dir/'DATA_SEMANTICS_STOP.json').exists():
        raise RuntimeError('DATA_SEMANTICS_STOP: this pilot is quarantined; do not resume or override its data')
