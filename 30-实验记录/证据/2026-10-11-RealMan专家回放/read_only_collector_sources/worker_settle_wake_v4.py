"""Frozen worker v2 private binding: select extra-settle collector only.

Defaults to the source-registering v9 thin expert; no GPU/worker starts on import.
"""
from __future__ import annotations
import ast
import copy
import sys
from pathlib import Path
import collector_settle_wake_v4 as collector


def private_worker_tree(payload):
    tree = ast.parse(payload)
    sites = 0
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'generation_runner_v2' and alias.asname == 'collector':
                    alias.name = 'collector_settle_wake_v4'
                    sites += 1
    if sites != 1:
        raise RuntimeError('Exactly one original worker collector import is required')
    ast.fix_missing_locations(tree)
    return tree


def worker_argv(argv):
    result = list(argv)
    selections = []
    for index, token in enumerate(result[1:], 1):
        if token == '--expert-module':
            if index + 1 >= len(result):
                raise ValueError('Missing expert module value')
            selections.append(result[index + 1])
        elif token.startswith('--expert-module='):
            selections.append(token.split('=', 1)[1])
    if not selections:
        result.extend(['--expert-module', 'expert_v9_settle_wake_v4'])
    elif len(selections) != 1 or selections[0] != 'expert_v9_settle_wake_v4':
        raise ValueError('Extra-settle worker requires exactly the pinned expert_v9_settle_wake_v4')
    return result


def main():
    original = Path(__file__).resolve().parent / 'generation_worker_v2.py'
    collector.require_source(original, collector.ORIGINAL_WORKER_SHA256)
    collector.producer_pins()
    tree = private_worker_tree(original.read_text(encoding='utf-8-sig'))
    arguments = sys.argv
    sys.argv = worker_argv(arguments)
    try:
        # Original parser, capture callback, execution, export, timeout and cleanup.
        namespace = {'__name__': '__main__', '__file__': str(original)}
        exec(compile(tree, str(Path(__file__).resolve()) + ':private_worker', 'exec'), namespace)
    finally:
        sys.argv = arguments


if __name__ == '__main__':
    main()