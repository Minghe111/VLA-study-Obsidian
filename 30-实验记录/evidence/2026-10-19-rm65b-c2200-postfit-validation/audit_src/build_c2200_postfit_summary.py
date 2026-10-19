"""CPU-only aggregation of completed frozen-C2200 validation receipts."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = {}
    files = []
    protected = []
    for name in ('heldout_val', 'paraphrase', 'counterfactual_corrected', 'revision_only'):
        path = args.reports/name/'report.json'
        report = read(path)
        assert report['status'] == 'completed'
        assert report['optimizer_updates'] == 0
        guard = report['parameter_protection']
        assert guard['all_parameters_exact'] and guard['checkpoint_unchanged']
        assert not guard['optimizer_created'] and guard['parameters_with_grad'] == 0
        protocol[name] = dict(
            count=report['evaluation_count'], strict=report['summary']['strict_passed'],
            eos=report['summary']['eos'],
            by_mode={mode: value['strict_passed'] for mode, value in report['by_mode'].items()},
            unique_inputs=report['unique_actual_actor_inputs'],
        )
        protected.append(dict(name=name, inference_parameter_and_source_guard_passed=True))
        files.append(receipt(path))
    legacy = args.reports/'counterfactual'
    files.extend(receipt(legacy/name) for name in ('report.json', 'SCHEMA_ERRATUM.json'))
    old_task = {}
    for name in ('old_task_baseline', 'old_task_c2200', 'old_task_baseline_k16', 'old_task_c2200_k16'):
        path = args.reports/name/'report.json'
        report = read(path)
        assert report['status'] == 'completed'
        assert report['optimizer_updates'] == 0 and report['training'] is False
        assert report['parameters_unchanged'] and report['parameter_versions_unchanged']
        assert report['no_parameter_gradients']
        episodes = report['episodes']
        assert all(row['status'] == 'completed' and row['workers_closed'] for row in episodes)
        old_task[name] = dict(
            K_exec=report['K_exec'], success=sum(bool(row['success']) for row in episodes),
            count=len(episodes), steps=[row['actual_control_steps'] for row in episodes],
            whole_card_sampled_peak_mib=report['sampled_whole_card_peak_mib'],
            task_retention_claim=False,
        )
        files.append(receipt(path))
    high_path = args.reports/'highlevel_closedloop/report.json'
    high = read(high_path)
    assert high['status'] == 'completed'  # orchestration finished; individual cases can fail
    assert high['all_parameters_unchanged'] and high['all_parameter_versions_unchanged']
    assert high['all_source_files_unchanged'] and high['gradients_after'] == 0
    assert high['optimizer_updates'] == 0 and high['mode'] == 'autonomous_high_level'
    cases = []
    for row in high['results']:
        assert row['workers_closed']
        cases.append(dict(
            case_id=row['case']['case_id'], status=row['status'],
            controls=row['executed_controls'],
            error=row['episode_metadata'].get('collection_error'),
            subtask_decisions=row['diagnosis']['subtask_decisions'],
            initial_role_correct=row['diagnosis']['subtask_first_role_matches'],
            sim_time_s=row['diagnosis']['sim_time_s'],
            physics_steps=row['diagnosis']['physics_step_count'],
        ))
    files.append(receipt(high_path))
    resources_path = args.reports/'highlevel_resource.json'
    resources = read(resources_path)
    assert resources['returncode'] == 0
    files.append(receipt(resources_path))
    audit_paths = [
        '/tmp/protocol_postfit_independent_audit.json',
        '/tmp/protocol_cf_revision_independent_audit.json',
        '/tmp/old_task_pair_independent_audit.json',
        '/tmp/old_task_k16_pair_independent_audit.json',
        '/tmp/highlevel_closedloop_independent_audit.json',
        '/tmp/highlevel_history_coverage_audit.json',
        '/tmp/c2200_final_source_protection.json',
    ]
    audits = {Path(path).name: receipt(path) for path in audit_paths}
    summary = dict(
        status='validation_completed_capability_gates_failed',
        checkpoint=str(high['checkpoint']), checkpoint_global_step=2200,
        optimizer_updates_this_validation=0, formal_3000_training=False,
        contract=high['contract'], runtime_MEM=False, cameras_RGB=3,
        protocol=protocol, old_task=old_task, highlevel_cases=cases,
        inference_parameter_protection=protected,
        legacy_counterfactual=dict(
            forty_SUBTASK_rows_invalid=True, overall_score_excluded=True,
            reason='Legacy evaluator used list actor_memory instead of the real nested tool schema; corrected separate run preserved.',
        ),
        capability_gates=dict(
            canonical_heldout_template=True, paraphrase_all_correct=False,
            crossed_goal_memory=False, revision_invariance=False,
            first_highlevel_tool_action_cycle=True, sustained_highlevel_loop=False,
            original_motor_success_baseline_established=False,
            task_skill_retention_verified=False, full_ordered_verify_return_success_verified=False,
            new_visual_grounding_verified=False, MEM_joint_training_verified=False,
        ),
        critical_findings=[
            'Heldout60 rows represent28 actual inputs and9 reused target strings, not60 new physical scenes.',
            'At fixed contexts all8 revision-only triples changed EMPTY/KEEP/CORRECT as1/2/4; local shortcut, not a global model characterization.',
            'Both highlevel cases executed32 targets after a valid first cycle, then strict READ rejected Sub right right.',
            'All fit96/train360/val60 history fields are empty lists; controller emits empty string initially and last-subtask text later. [] vs empty string changes actual prefix tokens. Confirmed serialization inconsistency plus missing dynamic-history coverage; not a sole-cause attribution.',
            'Original SFT and C2200 both0/2 at K1 and K16; no task preservation or LoRA-caused failure claim.',
            'Outer completed receipts describe completed orchestration and do not convert failed cases into successful tasks.',
        ],
        resource=dict(
            max_whole_card_sampled_peak_mib=max(
                resources['sampled_peak_mib'],
                *(row['whole_card_sampled_peak_mib'] for row in old_task.values())),
            scope='Sampled whole-card use including desktop/simulator, no exact peak or process attribution.',
        ),
        reports=files, independent_audit_and_provenance_receipts=audits,
        source_entries_replaced=False, human_or_expert_interventions=0,
        automatic_training_after_failure=False,
    )
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'], cases=cases, resource=summary['resource']), ensure_ascii=False))


if __name__ == '__main__':
    main()
