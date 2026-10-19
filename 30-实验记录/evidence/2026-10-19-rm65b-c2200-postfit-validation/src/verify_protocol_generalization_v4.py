"""Frozen C2200 protocol checks. CPU audit by default; generation requires --execute.

Val is a development split. Test inference remains closed until a completed
validation report with the same checkpoint/script contract is supplied.
Counterfactual targets below are formal static-role rules, not physical labels.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
SOURCE=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_sft_20261018_v1')
LABELS=SOURCE/'data/protocol_labels_reviewed_v1'
CHECKPOINT=ROOT/'reports/continuation_v1/language_warmup_96_2200/checkpoint'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*2**20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def target_rule(row):
    sides=re.findall(r'\b(left|right)\b',row['context']['goal'].lower())
    if not sides:
        raise ValueError('The explicit static goal must identify its first side')
    side=sides[0]
    fact=f'The first requested sample is initially on the {side}.'
    if row['mode']=='READ':
        target={'read_ids':['sample_identity']}
    elif row['mode']=='SUBTASK':
        return f'Pick up and hold the sample initially on the {side}.'
    else:
        slot=row['context']['selected_memory'][0]
        if slot['status']!='active':
            operation={'confirmed':True,'op':'WRITE','slot_id':'sample_identity','value':fact}
        elif slot['value']!=fact:
            operation={'confirmed':True,'op':'CORRECT','slot_id':'sample_identity','value':fact}
        elif slot['confirmed'] is not True:
            operation={'confirmed':True,'op':'KEEP','slot_id':'sample_identity'}
        else:
            operation=None
        target={'operations':[] if operation is None else [operation]}
    return json.dumps(target,sort_keys=True,separators=(',',':'))


def validate_target(row):
    expected=target_rule(row)
    if row['target_text']!=expected:
        raise ValueError(f'Target differs from actor-visible static role rule: {row["sample_id"]}')
    side=re.findall(r'\b(left|right)\b',row['context']['goal'].lower())[0]
    if row['evaluation']['target_side']!=side or row['evidence']['candidate_fact']!=f'The first requested sample is initially on the {side}.':
        raise ValueError('Target-side/evidence differs from the explicit goal')
    if row['metadata'].get('motor_labels') is not False:
        raise ValueError('Static protocol rows cannot establish robot motor outcomes')
    if row.get('decision_physics_step')!=15*row['decision_step']:
        raise ValueError('The60ms/15physicsstep source contract changed')
    if row['observed_until_step']>row['decision_step'] or row['evidence']['observed_until_step']>row['decision_step']:
        raise ValueError('Future observation leakage')
    return True


def historical_seen():
    seen=defaultdict(set)
    reports=[]
    for path in sorted((SOURCE/'reports/protocol_sft_sequence_v1').glob('*/report.json')):
        data=read(path)
        count=0
        for case in data.get('cases',[]):
            ref=case.get('source',{})
            if isinstance(ref,dict) and ref.get('path'):
                seen[Path(ref['path']).name].add(ref['row'])
                count+=1
        if count:
            reports.append({'path':str(path),'case_count':count,'status':data.get('status')})
    return dict(seen),reports


def summary(data):
    return {'rows':len(data),'modes':dict(Counter(x['mode'] for x in data)),
        'families':sorted({x['family_id'] for x in data}),
        'episode_count':len({x['episode_id'] for x in data}),
        'observation_files':len({x['observation']['npz_path'] for x in data}),
        'unique_actor_input_digests':len({x['actor_input_digest'] for x in data}),
        'unique_target_strings':len({x['target_text'] for x in data}),
        'write_operations':dict(Counter(('EMPTY' if not json.loads(x['target_text'])['operations'] else json.loads(x['target_text'])['operations'][0]['op'])
            for x in data if x['mode']=='WRITE'))}


def build_memory_condition(catalogue, memory_name):
    """Use the real strict memory tool, without inventing a current WRITE."""
    from rlinf.data.storage.episode_memory import EpisodeMemory
    bank=EpisodeMemory(catalogue,'counterfactual_eval_env','counterfactual_eval_episode',strict_protocol=True)
    ids=['sample_identity']
    bank.read(ids)
    if memory_name!='empty':
        prefix,side=memory_name.split('_',1)
        if prefix not in {'confirmed','unconfirmed'} or side not in {'left','right'}:
            raise ValueError('Unknown explicit counterfactual memory state')
        bank.apply({'op':'WRITE','slot_id':'sample_identity',
            'value':f'The first requested sample is initially on the {side}.',
            'confirmed':prefix=='confirmed'},source={'kind':'formal_evaluation_memory_seed','step':0})
    # Start a fresh tool round after seeding. No operation has been applied in
    # this round: actor_view must therefore expose current_writes=[].
    selected=bank.read(ids)
    view=bank.actor_view(ids,[])
    if view!={'selected':selected,'current_writes':[]}:
        raise ValueError('Native memory tool returned an unexpected public envelope')
    return bank,selected,view


def actor_visible_memory(row):
    context=row['context']
    return context.get('actor_memory',context.get('selected_memory',{}))


def validate_actor_visible_schema(row):
    """Validate actual model-visible fields, not just reference target text."""
    context,mode=row['context'],row['mode']
    from rlinf.data.storage.episode_memory import EpisodeMemory
    bank=EpisodeMemory(context['catalogue'],'schema_audit_env','schema_audit_episode',strict_protocol=True)
    if context['catalogue']!=bank.catalogue_view():
        raise ValueError('Catalogue differs from the native memory tool schema')
    visible=actor_visible_memory(row)
    def slots(values):
        if not isinstance(values,list):
            raise ValueError('Selected/current_writes must be a list of public memory slots')
        for slot in values:
            if not isinstance(slot,dict) or set(slot)!={'slot_id','status','value','confirmed','revision'}:
                raise ValueError('Memory slot fields differ from strict native actor lookup')
            if slot['slot_id']!='sample_identity' or type(slot['confirmed']) is not bool or type(slot['revision']) is not int:
                raise ValueError('Invalid public slot identifier/confirmed/revision types')
            if slot['status']=='empty':
                if slot['value'] is not None or slot['confirmed'] is not False or slot['revision']!=0:
                    raise ValueError('Malformed native empty slot')
            elif slot['status']=='active':
                if not isinstance(slot['value'],str) or not slot['value'] or slot['revision']<1:
                    raise ValueError('Malformed native active slot')
            else:
                raise ValueError('These controls permit only native empty/active slots')
    if mode=='READ':
        if visible!={}:
            raise ValueError('READ must receive catalogue without slot values')
    elif mode=='WRITE':
        if context.get('read_ids')!=['sample_identity'] or 'actor_memory' in context:
            raise ValueError('WRITE requires selected_memory after the actual READ')
        slots(visible)
        if len(visible)!=1:
            raise ValueError('One-slot control must expose one selected response')
    elif mode=='SUBTASK':
        if context.get('read_ids')!=['sample_identity'] or not isinstance(visible,dict) or set(visible)!={'selected','current_writes'}:
            raise ValueError('SUBTASK requires native actor_view selected/current_writes envelope plus read_ids')
        slots(visible['selected']);slots(visible['current_writes'])
        if len(visible['selected'])!=1:
            raise ValueError('SUBTASK must retain the one actual selected slot')
    else:
        raise ValueError('Unknown protocol mode')
    return True


def counterfactual_rows(validation,observation_limit=4,observation_signatures=None):
    # Spread small smoke tests over the four heldout episodes, rather than
    # selecting four adjacent frames from a single episode.
    by_episode=defaultdict(dict)
    for row in validation:
        by_episode[row['episode_id']].setdefault(row['observation']['npz_path'],row)
    ordered=[]
    lists=[list(sorted(v.values(),key=lambda x:x['decision_step'])) for _,v in sorted(by_episode.items())]
    for index in range(max(map(len,lists))):
        ordered.extend(v[index] for v in lists if index<len(v))
    if observation_signatures is not None:
        unique_observations={}
        for row in ordered:
            signature=observation_signatures[row['observation']['npz_path']]
            unique_observations.setdefault(signature,row)
        ordered=list(unique_observations.values())
    if observation_limit:
        ordered=ordered[:observation_limit]
    output=[]
    memory_names=('empty','confirmed_left','unconfirmed_left','confirmed_right','unconfirmed_right')
    for obs_index,base in enumerate(ordered):
        for goal_side in ('left','right'):
            opposite='right' if goal_side=='left' else 'left'
            goal=f'Verify the sample initially on the {goal_side} and return it home, then verify the sample initially on the {opposite} and return it home.'
            for mode in ('READ','WRITE','SUBTASK'):
                for memory_name in (('none',) if mode=='READ' else memory_names):
                    row=deepcopy(base)
                    context={'goal':goal,'catalogue':deepcopy(base['context']['catalogue']),'visible_history':[],'mode':mode}
                    if mode in {'WRITE','SUBTASK'}:
                        _,selected,view=build_memory_condition(context['catalogue'],memory_name)
                        context['read_ids']=['sample_identity']
                        if mode=='WRITE':
                            context['selected_memory']=selected
                        else:
                            context['actor_memory']=view
                    row.update(mode=mode,context=context,supervision_kind='formal_counterfactual_eval_only',
                        sample_id=f'cf_val_obs{obs_index:02d}:{mode}:{memory_name}:goal_{goal_side}',split='val')
                    row['metadata'].update(training_eligible=False,evaluation_only=True,
                        label_basis='formal_actor_visible_static_role_rule_not_new_physical_evidence')
                    row['provenance']={'kind':'text_counterfactual_evaluation','source_sample_id':base['sample_id'],
                        'source_observation':base['observation'],'new_physical_event_claimed':False}
                    row['evaluation'].update(target_side=goal_side,required_terms=[] if mode!='SUBTASK' else ['pick','hold'])
                    row['evidence']['candidate_fact']=f'The first requested sample is initially on the {goal_side}.'
                    row['counterfactual_group']={'observation_index':obs_index,'goal_side':goal_side,'memory':memory_name,
                        'goal_pair':f'obs{obs_index}:{mode}:{memory_name}',
                        'memory_pair':None if mode=='READ' else f'obs{obs_index}:{mode}:{goal_side}'}
                    row['target_text']=target_rule(row)
                    row['metadata']['memory_intervention_scope']='Controlled stale/empty/unconfirmed selected-memory stress; not asserted to be the result of a preceding correct WRITE'
                    validate_target(row)
                    validate_actor_visible_schema(row)
                    output.append(row)
    return output


def audit(config):
    from build_protocol_labels import _load_frame,actor_input_digest
    from rlinf.models.embodiment.openpi.recap_tokenizer import RecapTokenizer
    tokenizer=RecapTokenizer.from_local_paths(config['tokenizer']['text_tokenizer_path'],
        max_prefix_tokens=config['tokenizer']['max_prefix_tokens'],max_target_tokens=config['tokenizer']['max_target_tokens'])
    stats=read(config['model']['openpi_data']['norm_stats_path'])['norm_stats']['state']
    files={'fit96':ROOT/'data/fit_subsets_v1/fit96.train.jsonl',
        **{split:LABELS/f'{split}.reviewed.jsonl' for split in ('train','val','test')}}
    all_rows={split:rows(path) for split,path in files.items()}
    cache={}
    recalc={}
    for split,data in all_rows.items():
        for row in data:
            validate_target(row)
            validate_actor_visible_schema(row)
            path=row['observation']['npz_path']
            if path not in cache:
                cache[path]=_load_frame({'npz_path':path},stats)
            state,hashes=cache[path]
            digest=actor_input_digest(tokenizer,row,state,hashes['camera_pixel_sha256'])
            if digest!=row['actor_input_digest']:
                raise ValueError('Actual actor prefix/pixel digest differs from stored input identity')
            if row['evidence']['observation_refs'][0]['npz_sha256']!=hashes['npz_sha256']:
                raise ValueError('Raw source observation hash changed')
        recalc[split]=summary(data)
        recalc[split].update(path=str(files[split]),sha256=sha(files[split]),targets_rule_validated=True,
            actual_input_digest_recomputed=True)
    overlaps={}
    for a,b in (('fit96','val'),('fit96','test'),('train','val'),('train','test'),('val','test')):
        overlap={}
        for key in ('sample_id','family_id','episode_id','seed','actor_input_digest'):
            overlap[key]=len({x[key] for x in all_rows[a]}&{x[key] for x in all_rows[b]})
        overlap['observation_paths']=len({x['observation']['npz_path'] for x in all_rows[a]}&{x['observation']['npz_path'] for x in all_rows[b]})
        overlap['unique_target_strings']=len({x['target_text'] for x in all_rows[a]}&{x['target_text'] for x in all_rows[b]})
        overlap['actual_three_camera_pixel_triplets']=len({tuple(sorted(cache[x['observation']['npz_path']][1]['camera_pixel_sha256'].items())) for x in all_rows[a]}&
            {tuple(sorted(cache[x['observation']['npz_path']][1]['camera_pixel_sha256'].items())) for x in all_rows[b]})
        overlaps[f'{a}_vs_{b}']=overlap
    seen,reports=historical_seen()
    for split in ('val','test'):
        indices=seen.get(f'{split}.reviewed.jsonl',set())
        recalc[split]['historically_inferred_rows']=len(indices)
        recalc[split]['no_historical_inference_found_rows']=len(all_rows[split])-len(indices)
        recalc[split]['historically_inferred_indices']=sorted(indices)
        seen_digests={all_rows[split][i]['actor_input_digest'] for i in indices}
        fresh=[row for row in all_rows[split] if row['actor_input_digest'] not in seen_digests]
        recalc[split]['historically_inferred_unique_actor_inputs']=len(seen_digests)
        recalc[split]['fresh_actor_input_rows_in_searched_history']=len(fresh)
        recalc[split]['fresh_unique_actor_inputs_in_searched_history']=len({row['actor_input_digest'] for row in fresh})
        recalc[split]['not_inferred_rows_with_already_seen_actor_input']=sum(row['actor_input_digest'] in seen_digests for i,row in enumerate(all_rows[split]) if i not in indices)
    return {'status':'CPU_audit_passed_no_model_loaded','splits':recalc,'overlaps':overlaps,
        'historical_report_search_scope':'SOURCE/reports/protocol_sft_sequence_v1/*/report.json; absence is bounded to this search',
        'historical_inference_reports':reports,'test_read_for_source_exclusion_and_label_integrity_only':True,
        'semantic_limit':'All labels are static role/memory rules from authored contexts; nine target strings, not visual grounding or robot task success',
        'test_policy':'Use val for interface decisions; no test inference until a frozen validation report for the same C2200 and evaluator hash'},all_rows,cache,tokenizer,stats


def score_predictions(cases):
    result={'count':len(cases)}
    for key in ('eos','nonempty','json','schema_legal','target_match','side_binding_lexical','text_exact','semantic_passed','strict_passed','strict_text_exact'):
        values=[x['checks'][key] for x in cases if key in x['checks']]
        result[key]={'passed':sum(values),'eligible':len(values)}
    result['all_strict_passed']=bool(cases) and all(x['checks']['strict_passed'] for x in cases)
    return result


def run_generation(actor,records,output):
    import torch
    from protocol_pilot import check_output,write_json
    from rlinf.data.datasets.recap.hierarchical_dataset import load_observation
    from rlinf.models.embodiment.openpi.recap_tokenizer import canonical_json
    cases=[]
    for row in records:
        context=row['context']
        with torch.no_grad():
            prediction=actor.generate(load_observation(row['observation']),row['mode'],{
                'goal':context['goal'],'memory_index':context['catalogue'],
                'history':canonical_json(context['visible_history']),
                'visible_memory':context.get('actor_memory',context.get('selected_memory',{}))})
        checks=check_output(row,prediction.text,prediction.ended_with_eos)
        checks['text_exact']=prediction.text.strip()==row['target_text'].strip()
        checks['semantic_passed']=checks['passed']
        checks['strict_passed']=checks['passed'] and (row['mode']!='SUBTASK' or checks['text_exact'])
        checks['strict_text_exact']=checks['eos'] and checks['text_exact']
        cases.append({'sample_id':row['sample_id'],'actor_input_digest':row.get('actor_input_digest'),'mode':row['mode'],'target_text':row['target_text'],
            'text':prediction.text,'token_ids':prediction.token_ids,'checks':checks,
            'counterfactual_group':row.get('counterfactual_group')})
        write_json(output,{'status':'running','cases':cases})
    return cases


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','val','counterfactual','test'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    parser.add_argument('--observations',type=int,choices=[4,20],default=4)
    parser.add_argument('--frozen-validation-report',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'code/RLinf')]
    import yaml
    cfg=yaml.safe_load((args.checkpoint.parent/'effective_config.yaml').read_text())
    audit_result,all_rows,cache,tokenizer,stats=audit(cfg)
    if args.stage=='counterfactual':
        from build_protocol_labels import actor_input_digest
        signatures={}
        for row in all_rows['val']:
            probe=deepcopy(row)
            probe['mode']='READ'
            probe['context']={'goal':'Verify the sample initially on the left and return it home, then verify the sample initially on the right and return it home.',
                'catalogue':deepcopy(row['context']['catalogue']),'visible_history':[],'mode':'READ'}
            normalized,hashes=cache[row['observation']['npz_path']]
            signatures[row['observation']['npz_path']]=actor_input_digest(tokenizer,probe,normalized,hashes['camera_pixel_sha256'])
        data=counterfactual_rows(all_rows['val'],args.observations,observation_signatures=signatures)
        seen={}
        for row in data:
            state,hashes=cache[row['observation']['npz_path']]
            row['actor_input_digest']=actor_input_digest(tokenizer,row,state,hashes['camera_pixel_sha256'])
            previous=seen.setdefault(row['actor_input_digest'],row['target_text'])
            if previous!=row['target_text']:
                raise ValueError('Conflicting targets for identical actual counterfactual actor inputs')
        audit_result['counterfactual']=summary(data)
        audit_result['counterfactual']['target_basis']='Formal visible-context rule only, no fabricated physical event'
        audit_result['counterfactual']['actual_visible_memory_schema_validated']=True
        audit_result['counterfactual']['memory_tool']='EpisodeMemory(strict_protocol=True).read/actor_view; current_writes empty in seeded fresh round'
        audit_result['counterfactual']['subtask_scope']='Stale/empty/unconfirmed selected-memory pressure, not a claimed post-correct-WRITE natural trajectory'
        audit_result['counterfactual']['v2_erratum']='Prior v2 SUBTASK inputs were list-shaped, so their40 failures do not establish model generalization failure; prior WRITE16/40 remains a valid failure result'
        audit_result['counterfactual']['source_observation_paths']=len(signatures)
        audit_result['counterfactual']['available_unique_image_and_state_inputs']=len(set(signatures.values()))
        audit_result['counterfactual']['selection']='Deduplicate actual image+normalized-state-prefix before choosing observations'
        audit_result['counterfactual']['unique_goal_pairs']=len({x['counterfactual_group']['goal_pair'] for x in data})
        audit_result['counterfactual']['unique_memory_groups']=len({x['counterfactual_group']['memory_pair'] for x in data if x['counterfactual_group']['memory_pair']})
        audit_result['counterfactual']['original_val_actor_input_overlap']=len({x['actor_input_digest'] for x in data}&{x['actor_input_digest'] for x in all_rows['val']})
        audit_result['counterfactual']['fit96_input_digest_overlap']=len({x['actor_input_digest'] for x in data}&{x['actor_input_digest'] for x in all_rows['fit96']})
    else:
        data=all_rows.get(args.stage,[])
    plan={'status':'planned','stage':args.stage,'checkpoint':str(args.checkpoint),
        'script_sha256':sha(Path(__file__)),'evaluation_count':len(data),'audit':audit_result,
        'optimizer_updates':0,'simulation_rollouts':0,'decoding':'NativePi05Actor unrestricted greedy, max96, real EOS; no repair/grammar constraints'}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'plan_and_audit.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    if args.stage=='audit' or not args.execute:
        print(json.dumps({'status':'CPU_audit_only_no_model_loaded','rows':len(data),'output':str(args.output)}))
        return plan
    if args.stage=='test':
        if args.frozen_validation_report is None:
            raise ValueError('Test remains closed until a frozen completed validation report is supplied')
        frozen=read(args.frozen_validation_report)
        if (frozen.get('status')!='completed' or frozen.get('stage')!='val'
            or frozen.get('checkpoint')!=str(args.checkpoint) or frozen.get('script_sha256')!=sha(Path(__file__))):
            raise ValueError('Frozen validation/checkpoint/evaluator contract differs')
    from rlinf.runners.hierarchical_recap_runner import NativeHierarchicalBackend
    version=read(args.checkpoint/'version.json')
    if version['global_step']!=2200 or version['training_manifest']['training_stage']!='language_warmup':
        raise ValueError('Only the frozen C2200 language stage may be evaluated here')
    cfg['model']['model_path']=str(args.checkpoint)
    from protocol_pilot import parameter_digest
    checkpoint_weights_sha_before=sha(args.checkpoint/'full_weights.pt')
    actor=NativeHierarchicalBackend(cfg)._actor()
    actor.model.eval()
    parameter_guard_before={str(trainable):parameter_digest(actor.model,trainable=trainable) for trainable in (False,True)}
    cases=run_generation(actor,data,args.output/'generation_progress.json')
    parameter_guard_after={str(trainable):parameter_digest(actor.model,trainable=trainable) for trainable in (False,True)}
    checkpoint_weights_sha_after=sha(args.checkpoint/'full_weights.pt')
    if parameter_guard_before!=parameter_guard_after or checkpoint_weights_sha_before!=checkpoint_weights_sha_after:
        raise RuntimeError('Read-only model/checkpoint parameter protection failed')
    final={**plan,'status':'completed','finished_utc':datetime.now(timezone.utc).isoformat(),'cases':cases,
        'summary':score_predictions(cases),'by_mode':{mode:score_predictions([x for x in cases if x['mode']==mode]) for mode in ('READ','WRITE','SUBTASK')},
        'scope':'Heldout validation or controlled evaluation-only protocol conditions; not a training fitting report, visual grounding or task SR result',
        'parameter_protection':{'before':parameter_guard_before,'after':parameter_guard_after,'all_parameters_exact':True,
            'checkpoint_weights_sha_before':checkpoint_weights_sha_before,'checkpoint_weights_sha_after':checkpoint_weights_sha_after,
            'checkpoint_version_sha256':sha(args.checkpoint/'version.json'),'checkpoint_unchanged':True,
            'optimizer_created':False,'parameters_with_grad':sum(p.grad is not None for p in actor.model.parameters())}}
    deduplicated={}
    for case in cases:
        deduplicated.setdefault(case['actor_input_digest'],case)
    unique_cases=list(deduplicated.values())
    final['unique_actual_actor_inputs']={'count':len(unique_cases),'summary':score_predictions(unique_cases),
        'by_mode':{mode:score_predictions([x for x in unique_cases if x['mode']==mode]) for mode in ('READ','WRITE','SUBTASK')}}
    if args.stage=='val':
        old=set(audit_result['splits']['val']['historically_inferred_indices'])
        final['historically_seen']=score_predictions([x for i,x in enumerate(cases) if i in old])
        final['not_previously_inferred_in_searched_reports']=score_predictions([x for i,x in enumerate(cases) if i not in old])
        old_digests={all_rows['val'][i]['actor_input_digest'] for i in old}
        final['history_by_actual_actor_input']={
            'historically_seen_unique_inputs':score_predictions([x for x in unique_cases if x['actor_input_digest'] in old_digests]),
            'fresh_unique_inputs_in_searched_history':score_predictions([x for x in unique_cases if x['actor_input_digest'] not in old_digests]),
            'historically_seen_input_rows':score_predictions([x for x in cases if x['actor_input_digest'] in old_digests]),
            'fresh_input_rows_in_searched_history':score_predictions([x for x in cases if x['actor_input_digest'] not in old_digests])}
    if args.stage=='counterfactual':
        pair_results={}
        for kind in ('goal_pair','memory_pair'):
            groups=defaultdict(list)
            for case in cases:
                key=case['counterfactual_group'].get(kind)
                if key is not None:
                    groups[key].append(case)
            pair_results[kind]={'count':len(groups),'all_cases_passed':sum(all(x['checks']['strict_passed'] for x in group) for group in groups.values()),
                'groups':[{'pair_id':key,'count':len(group),'all_passed':all(x['checks']['strict_passed'] for x in group),
                    'expected_distinct_outputs':len({x['target_text'] for x in group}),
                    'observed_distinct_outputs':len({x['text'] for x in group})} for key,group in groups.items()]}
        final['controlled_pairs']=pair_results
    (args.output/'report.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
    return final


if __name__=='__main__':
    main()
