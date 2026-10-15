from recompute_v12_wrench_provenance import recompute_v12_wrench_provenance
wrench_recomputed=recompute_v12_wrench_provenance(task,OUT)
print('WRENCH_RECOMPUTED_RESULT',wrench_recomputed['passed'],wrench_recomputed['max_force_error_N'],wrench_recomputed['max_torque_error_Nm'],flush=True)
assert wrench_recomputed['passed'],wrench_recomputed.get('error')
