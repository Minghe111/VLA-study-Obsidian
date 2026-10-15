assert json.loads((OUT/'actual36_geometry_audit.json').read_text())['passed']
assert json.loads((OUT/'wrench_provenance_recomputed_20261003.json').read_text())['passed']
assert sha(BASE/'accepted_reset_replay.py')=='5ad50f270b99947e81476b6d5364628dd3e64fda758682b085b41cb8362832cb'
accepted_replay_version='v12'
exec(compile((BASE/'accepted_reset_replay.py').read_text(),str(BASE/'accepted_reset_replay.py'),'exec'),globals())
