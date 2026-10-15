"""Only the two V6 release-script edits needed for dedicated v1 evidence."""


def _once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Hybrid release site count differs: ' + repr(old[:100]))
    return text.replace(old, new, 1)


def patch_quality(text):
    text = _once(text, 'import streaming_export_staged as exporter\n',
                 'import streaming_export_staged as exporter\nimport hybrid_release_contract as hybrid_release\n')
    text = _once(text, "limits.get('MemorySwapMax') == '0'",
                 "limits.get('MemorySwapMax') == 'not-enforceable-on-v1'")
    text = _once(text,
        "    return {'passed': True, 'receipt_path': str(path.resolve()), 'receipt_sha256': sha(path),\n"
        "            'candidate_directory': str(candidate), 'entry': str(entry), 'entry_sha256': sha(entry),",
        "    hybrid_release.validate_stage(receipt, path, 'collect', HERE)\n"
        "    return {'passed': True, 'receipt_path': str(path.resolve()), 'receipt_sha256': sha(path),\n"
        "            'candidate_directory': str(candidate), 'entry': str(entry), 'entry_sha256': sha(entry),")
    compile(text, 'staged_quality.py', 'exec')
    return text


def patch_publisher(text):
    text = _once(text, 'import h5py\n',
                 'import h5py\nimport hybrid_release_contract as hybrid_release\n')
    text = _once(text, "limits.get('MemorySwapMax') == '0'",
                 "limits.get('MemorySwapMax') == 'not-enforceable-on-v1'")
    begin = "    readback = read_json(path.parent / 'cgroup_readback.json')\n"
    end = "    return {'receipt_path': str(path.resolve()), 'receipt_sha256': sha(path), 'passed': True,\n"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError('Unique v2 publisher readback/return site not found')
    a, b = text.index(begin), text.index(end)
    if a >= b:
        raise ValueError('Publisher readback must precede return')
    text = text[:a] + '    hybrid_release.validate_stage(receipt, path, stage, HERE)\n' + text[b:]
    compile(text, 'publish_staged_patch.py', 'exec')
    return text


def patch_render(text):
    """Describe the server's v1 guard without claiming swap was disabled."""
    text = _once(text,
        'Run this under run_with_resource_guard.py --phase render. It constructs no',
        'Run this under run_guarded_stage.py --stage render --run. It constructs no')
    text = _once(text,
        "'memory_max_GiB': 3, 'CPU_quota_cores': 2, 'swap_max_bytes': 0,",
        "'memory_max_GiB': 3, 'CPU_quota_cores': 2,\n"
        "            'swap_hard_disabled': False,\n"
        "            'zero_swap_evidence_required_from_external_guard': True,")
    compile(text, 'render_staged_video.py', 'exec')
    return text
