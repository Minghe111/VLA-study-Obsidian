#!/usr/bin/env python3
"""Isolated adapter for RoboTwin's original expert-code-generation loop.

No model response is invented or replayed. `generate` requires a real,
user-configured OpenAI-compatible API. `prompt` exercises the original
generate_code function up to its LLM boundary and explicitly stops there.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from types import MethodType
from datetime import datetime, timezone
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


@contextlib.contextmanager
def cwd(path):
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


def load_config(path):
    cfg = json.loads(Path(path).read_text()) if path else {}
    cfg.setdefault('source_root', str(DEFAULT_SOURCE))
    cfg.setdefault('objects_root', str(ROOT / 'metadata/assets/objects'))
    cfg.setdefault('api_key_env', 'OPENAI_API_KEY')
    cfg.setdefault('model_env', 'OPENAI_MODEL')
    cfg.setdefault('base_url_env', 'OPENAI_BASE_URL')
    cfg.setdefault('check_num', 10)
    return cfg


def model_settings(cfg):
    key_env = cfg['api_key_env']
    key = os.environ.get(key_env, '')
    model = cfg.get('model') or os.environ.get(cfg['model_env'], '')
    base = cfg.get('base_url') or os.environ.get(cfg['base_url_env'], '')
    if 'api_key' in cfg:
        raise ValueError('Do not put API secrets in JSON; configure api_key_env instead.')
    public = {
        'api_key_env': key_env, 'api_key_present': bool(key),
        'model': model or None, 'base_url_hostname': urlsplit(base).hostname if base else None,
        'ready': bool(key and model),
        'credentials_read_scope': 'named environment variable only',
    }
    return key, model, base, public


def official_modules(cfg):
    source = Path(cfg['source_root']).resolve()
    sys.path[:0] = [str(source / 'code_gen'), str(source)]
    # Official envs._GLOBAL_CONFIGS opens a relative asset list at import time.
    # Only import from the original cwd; all generated outputs use a fresh run cwd.
    with cwd(source):
        generation = importlib.import_module('task_generation')
        testing = importlib.import_module('test_gen_code')
    testing.OBJECTS_PATH = str(Path(cfg['objects_root']).resolve())
    extensions = cfg.get('additional_available_env_functions', {})
    if not isinstance(extensions, dict):
        raise ValueError('additional_available_env_functions must be a dictionary of explicit API descriptions.')
    generation.AVAILABLE_ENV_FUNCTION = dict(generation.AVAILABLE_ENV_FUNCTION)
    generation.AVAILABLE_ENV_FUNCTION.update(extensions)
    # Keep the original BASIC_INFO and all original examples. Added APIs are
    # explicitly described as local embodiment adapters, never official APIs.
    if extensions:
        generation.AVAILABLE_ENV_FUNCTION['LOCAL_EXTENSION_SCOPE'] = (
            'The following added entries are local RealMan SDK24 embodiment-adapter APIs; '
            'they are not part of the original RoboTwin API: ' + ', '.join(extensions))
    return generation, testing


def source_audit(cfg):
    source = Path(cfg['source_root'])
    files = ['code_gen/task_generation.py', 'code_gen/test_gen_code.py',
             'code_gen/gpt_agent.py', 'code_gen/prompt.py']
    placeholders = {}
    tree = ast.parse((source / 'code_gen/gpt_agent.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith('_api'):
                    val = node.value.value
                    placeholders[target.id] = {'is_placeholder': str(val).strip().lower() in
                                               {'your key', 'your_key', '', 'xxx'}}
    return {'files': {f: sha(source / f) for f in files}, 'official_key_placeholders': placeholders}


def preflight(cfg, output):
    _, _, _, public = model_settings(cfg)
    if cfg.get('task_module_path') and Path(cfg['task_module_path']).is_file():
        register_task_module(cfg, cfg.get('task_name', 'labware_pick_place'))
    generation, testing = official_modules(cfg)
    assert_runtime_contract(cfg)
    report = {
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': cfg['source_root'], 'source': source_audit(cfg),
        'actual_imported_runtime_modules': runtime_module_audit(),
        'llm': public, 'official_import_passed': bool(generation and testing),
        'official_functions': ['task_generation.main', 'task_generation.generate_code',
                               'test_gen_code.enrich_actors', 'test_gen_code.run'],
        'overlay_changes': ['LLM transport only', 'OBJECTS_PATH',
                            'generated-output cwd', 'optional task setup factory', 'check_num',
                            'explicitly labelled local embodiment API extensions'],
        'additional_available_env_functions': list(cfg.get('additional_available_env_functions', {})),
        'check_num': int(cfg['check_num']), 'feedback_attempts': 5,
        'official_success_threshold': 0.5,
        'actual_llm_request_sent': False, 'generated_expert_executed': False,
        'api_auto_loop_status': 'ready_for_configured_task' if public['ready'] else 'missing_llm_configuration',
        'session_bridge_available': True,
        'session_bridge_authorization': 'User explicitly authorized using the current Codex session on 2026-10-22.',
        'status': 'session_bridge_available' if not public['ready'] else 'api_and_session_modes_available',
    }
    dump(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if public['ready'] or cfg.get('generation_transport') == 'current_session' else 2


def new_run(label):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = ROOT / 'agent_runs' / f'{stamp}_{label}'
    path.mkdir(parents=True, exist_ok=False)
    (path / 'envs_gen').mkdir()
    (path / 'envs_gen/__init__.py').write_text('')
    return path


def load_factory(spec):
    file_part, sep, name = spec.rpartition(':')
    if not sep:
        raise ValueError('setup_factory must be /absolute/file.py:function or module:function')
    if file_part.endswith('.py'):
        path = Path(file_part).resolve()
        module_spec = importlib.util.spec_from_file_location('labware_agent_setup_adapter', path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(file_part)
    return getattr(module, name)


def register_task_module(cfg, task_name):
    if not cfg.get('task_module_path'):
        return
    path = Path(cfg['task_module_path']).resolve()
    name = f'envs.{task_name}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_module_audit():
    names = ['envs', 'envs._base_task', 'envs.utils', 'test_gen_code', 'task_generation']
    result = {}
    for name in names:
        module = sys.modules.get(name)
        source = getattr(module, '__file__', None)
        if source:
            path = Path(source).resolve()
            result[name] = {'file': str(path), 'sha256': sha(path)}
    return result


def assert_runtime_contract(cfg):
    expected = cfg.get('expected_base_task_source')
    if expected:
        actual = runtime_module_audit().get('envs._base_task', {}).get('file')
        if actual != str(Path(expected).resolve()):
            raise RuntimeError('Base_Task source differs from the explicitly selected RealMan runtime contract.')


def prepare_prompt(cfg, task_info_path, session_mode=False, candidate_path=None):
    task_info = json.loads(Path(task_info_path).read_text())
    if cfg.get('task_module_path') and Path(cfg['task_module_path']).is_file():
        register_task_module(cfg, task_info['task_name'])
    generation, testing = official_modules(cfg)
    assert_runtime_contract(cfg)
    run_dir = new_run('session_export' if session_mode else 'prompt_only')
    captured = {}

    class PromptCaptured(Exception):
        pass

    def capture(message, **kwargs):
        captured['messages'] = message
        captured['official_generate_kwargs'] = kwargs
        raise PromptCaptured()

    before = source_audit(cfg)
    generation.generate = capture
    with cwd(run_dir):
        try:
            generation.generate_code(task_info)
        except PromptCaptured:
            pass
    assert captured.get('messages'), 'Original generate_code did not reach the LLM boundary.'
    enriched = testing.enrich_actors(task_info['actor_list'])
    actor_checks = {}
    for name, info in task_info['actor_list'].items():
        if info.get('modelname') and info['modelname'] != 'None':
            entry = enriched[name]
            actor_checks[name] = {
                'functional_semantics_present': bool(entry.get('functional_points')),
                'contact_semantics_present': bool(entry.get('contact_points')),
                'modelname_removed_by_original_function': 'modelname' not in entry,
            }
            assert all(actor_checks[name].values()), f'Missing Agent point semantics for {name}'
    skeleton = run_dir / 'envs_gen' / f"gpt_{task_info['task_name']}.py"
    ast.parse(skeleton.read_text())
    dump(run_dir / 'prompt.json', captured)
    dump(run_dir / 'enriched_actors.json', enriched)
    dump(run_dir / 'prompt_verification.json', {
        'passed': True, 'actual_original_generate_code_called': True,
        'actual_original_enrich_actors_called': True,
        'actual_llm_request_sent': False, 'generated_expert_executed': False,
        'source_unchanged': before == source_audit(cfg),
        'additional_available_env_functions': list(cfg.get('additional_available_env_functions', {})),
        'actor_semantics_checks': actor_checks,
        'official_empty_skeleton_syntax_passed': True,
        'task_name': task_info['task_name'],
        'note': 'Stopped exactly at the LLM boundary; this is not a generated-expert success.'})
    dump(ROOT / 'agent_runs/latest_prompt.json', {'run_dir': str(run_dir), 'task_name': task_info['task_name'],
        'actual_llm_request_sent': False, 'generated_expert_executed': False})
    if session_mode:
        state = {
            'schema': 'robotwin-current-session-agent-v1', 'task_info': task_info,
            'generation_transport': 'current Codex session; explicitly authorized by the user',
            'remote_api_auto_loop': False, 'actual_llm_request_sent': False,
            'source_at_export': before, 'maximum_revisions': 5, 'attempts': [],
            'latest_messages_path': str(run_dir / 'prompt.json'),
            'status': 'waiting_for_current_session_candidate'}
        if candidate_path:
            candidate = Path(candidate_path).resolve()
            ast.parse(candidate.read_text())
            state['initial_candidate_registration'] = {
                'source_path': str(candidate), 'sha256': sha(candidate),
                'provenance': 'candidate supplied by the current Codex session; user-authorized',
                'physically_tested': False}
        dump(run_dir / 'session_state.json', state)
        dump(run_dir / 'session_export_receipt.json', {
            'actual_original_generate_code_called': True,
            'prompt_sha256': sha(run_dir / 'prompt.json'),
            'task_info_sha256': sha(task_info_path),
            'candidate_registration': state.get('initial_candidate_registration'),
            'actual_llm_request_sent': False, 'generated_expert_executed': False,
            'generation_transport': 'current Codex session bridge, not a remote API auto loop'})
        dump(ROOT / 'agent_runs/latest_session.json', {'session_dir': str(run_dir),
            'task_name': task_info['task_name'], 'status': 'waiting_for_current_session_candidate'})
    print(str(run_dir))
    return 0


def session_test(cfg, session_dir, candidate_path, check_num=None):
    """Run a real current-session candidate through the original physical tester.

    This never sends an API request and never claims the candidate was generated
    by a remote gpt_agent endpoint. Revision decisions remain in this session.
    """
    session_dir = Path(session_dir).resolve()
    session_dir.relative_to((ROOT / 'agent_runs').resolve())
    state_path = session_dir / 'session_state.json'
    state = json.loads(state_path.read_text())
    if state.get('schema') != 'robotwin-current-session-agent-v1':
        raise RuntimeError('Expected a session-export directory.')
    if len(state['attempts']) >= 5:
        raise RuntimeError('The original five-attempt budget has been reached; inspect preserved feedback.')
    if not cfg.get('setup_factory'):
        raise RuntimeError('Session testing requires an isolated task setup_factory.')
    candidate = Path(candidate_path).resolve()
    if not candidate.is_file() or candidate.suffix != '.py':
        raise RuntimeError('Candidate must be an existing Python source file.')
    candidate_text = candidate.read_text()
    registered = state.get('initial_candidate_registration')
    if not state['attempts'] and registered and sha(candidate) != registered['sha256']:
        raise RuntimeError('The first candidate differs from the SHA bound at session-export.')
    tree = ast.parse(candidate_text)
    task_info = state['task_info']
    task_name = task_info['task_name']
    class_name = f'gpt_{task_name}'
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1 or not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                                  n.name == 'play_once' for n in classes[0].body):
        raise RuntimeError('Candidate must define the expected generated subclass and play_once method.')
    ordinal = len(state['attempts']) + 1
    run_dir = session_dir / f'revision_{ordinal:02d}'
    run_dir.mkdir(exist_ok=False)
    (run_dir / 'envs_gen').mkdir()
    (run_dir / 'envs_gen/__init__.py').write_text('')
    frozen_candidate = run_dir / 'envs_gen' / f'{class_name}.py'
    shutil.copyfile(candidate, frozen_candidate)
    register_task_module(cfg, task_name)
    setup = load_factory(cfg['setup_factory'])
    generation, testing = official_modules(cfg)
    assert_runtime_contract(cfg)
    sys.path.insert(0, str(run_dir))
    for name in list(sys.modules):
        if name == 'envs_gen' or name.startswith('envs_gen.'):
            del sys.modules[name]
    importlib.invalidate_caches()
    check_num = int(cfg['check_num'] if check_num is None else check_num)
    if check_num < 1:
        raise ValueError('check_num must be positive.')
    before = source_audit(cfg)
    receipt = {
        'revision': ordinal, 'task_name': task_name,
        'generation_transport': 'current Codex session; user-authorized',
        'candidate_source_path': str(candidate), 'frozen_candidate': str(frozen_candidate),
        'candidate_sha256': sha(frozen_candidate), 'check_num': check_num,
        'remote_api_auto_loop': False, 'actual_llm_request_sent': False,
        'original_test_gen_code_run_called': False,
        'actual_imported_runtime_modules': runtime_module_audit(),
        'task_module_sha256': sha(cfg['task_module_path']),
        'source': before, 'status': 'starting_physical_test',
        'note': 'Current-session generation plus original RoboTwin physical feedback; not an API automatic loop.'}
    task_sources = {name: sha(ROOT / name) for name in ('labware_task.py', 'robot_expert.py')
                    if (ROOT / name).is_file()}
    receipt['task_implementation_sha256'] = task_sources
    dump(run_dir / 'session_receipt.json', receipt)
    try:
        task, args = setup(task_name=task_name, runtime_config=cfg, run_dir=str(run_dir))
        if task.__class__.__name__ != class_name or task.__class__.__module__ != f'envs_gen.{class_name}':
            raise RuntimeError('Factory selected a reference fallback instead of the current-session candidate.')
        # Preserve original run verbatim while isolating its repeated setup_demo
        # calls. Otherwise all ten seeds would share the factory's runtime_path.
        original_setup = task.setup_demo
        original_close = task.close
        original_check = task.check_success
        trial = {}

        def scoped_setup(this, **kwargs):
            seed = int(kwargs['seed'])
            path = run_dir / 'simulation' / f'seed_{seed:04d}'
            path.mkdir(parents=True, exist_ok=False)
            kwargs['runtime_path'] = str(path)
            trial.clear()
            trial.update(seed=seed, path=path, check_success_returns=[])
            dump(path / 'trial_status.json', {'seed': seed, 'status': 'initializing'})
            result = original_setup(**kwargs)
            dump(path / 'trial_status.json', {'seed': seed, 'status': 'running',
                'capture_rgb': bool(this.capture_rgb), 'center_xy': list(this.center_xy), 'yaw': this.yaw})
            return result

        def scoped_check(this):
            result = original_check()
            trial['check_success_returns'].append(bool(result))
            return result

        def scoped_close(this):
            if trial.get('path'):
                path = trial['path']
                result = {'seed': trial['seed'], 'status': 'complete_or_original_run_exception',
                    'check_success_returns': trial['check_success_returns'],
                    'plan_success': bool(getattr(this, 'plan_success', False)),
                    'metrics': getattr(this, 'metrics', {}),
                    'physics_tick_count': getattr(this, 'physics_step_count', None),
                    'transition_count': len(getattr(this, 'transition_log', [])),
                    'capture_rgb': bool(getattr(this, 'capture_rgb', False)),
                    'center_xy': list(getattr(this, 'center_xy', []) or []),
                    'yaw': getattr(this, 'yaw', None), 'hardware_io': False,
                    'task_implementation_sha256': task_sources}
                bottle = getattr(this, 'bottle', None)
                if bottle is not None:
                    result['final_object_pose'] = bottle.get_pose().to_transformation_matrix().tolist()
                dump(path / 'trial_result.json', result)
                dump(path / 'physics_trace.json', getattr(this, 'physics_trace', []))
                dump(path / 'transitions.json', getattr(this, 'transition_log', []))
                dump(path / 'trial_status.json', {'seed': trial['seed'], 'status': 'closed',
                    'success': bool(result['metrics'].get('success', False))})
            return original_close()

        task.setup_demo = MethodType(scoped_setup, task)
        task.check_success = MethodType(scoped_check, task)
        task.close = MethodType(scoped_close, task)
        receipt['original_test_gen_code_run_called'] = True
        dump(run_dir / 'session_receipt.json', receipt)
        # Original tester only reads source-cwd assets; the task factory routes
        # episode outputs explicitly into this revision's simulation directory.
        with (run_dir / 'official_test_console.log').open('w', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), cwd(cfg['source_root']):
                rate, error, error_count, records = testing.run(task, args, check_num=check_num)
        receipt.update({'success_rate': rate, 'error_message': error, 'error_count': error_count,
            'run_records': records, 'official_threshold_0_5_met': rate >= 0.5,
            'all_trials_passed': rate == 1.0, 'status': 'physical_test_complete'})
    except Exception:
        receipt.update({'status': 'physical_test_error', 'exception_trace': traceback.format_exc()})
        rate, error, error_count, records = 0.0, receipt['exception_trace'], check_num, []
    receipt['source_unchanged'] = before == source_audit(cfg)
    receipt['task_implementation_unchanged_during_test'] = task_sources == {
        name: sha(ROOT / name) for name in task_sources}
    receipt['candidate_unchanged'] = sha(frozen_candidate) == receipt['candidate_sha256']
    dump(run_dir / 'session_receipt.json', receipt)

    # Keep the candidate in conversation history, then build the feedback user
    # prompt by actually calling original generate_code's error branch.
    prior = json.loads(Path(state['latest_messages_path']).read_text())
    messages = prior['messages']
    messages.append({'role': 'assistant', 'content': '```python\n' + candidate_text + '\n```'})
    captured = {}
    class FeedbackCaptured(Exception):
        pass
    def capture(message, **kwargs):
        captured['messages'] = message
        captured['official_generate_kwargs'] = kwargs
        raise FeedbackCaptured()
    if rate == 1.0:
        # Original run returns a default error label even when error_count=0.
        # Do not feed its "unsuccessful" generation branch a successful trial.
        captured = {'messages': messages, 'official_error_prompt_called': False,
            'physical_feedback': f'All {check_num} actual physical trials passed. No revision requested.'}
    else:
        generation.generate = capture
        feedback_info = dict(task_info)
        feedback_info['current_code'] = candidate_text
        with cwd(run_dir):
            try:
                generation.generate_code(feedback_info, las_error=error, message=messages)
            except FeedbackCaptured:
                pass
        captured['official_error_prompt_called'] = True
    assert captured.get('messages'), 'Original generate_code did not produce a feedback prompt.'
    dump(run_dir / 'feedback_prompt.json', captured)
    state['attempts'].append({'revision': ordinal, 'receipt_path': str(run_dir / 'session_receipt.json'),
        'success_rate': rate, 'status': receipt['status'], 'candidate_sha256': receipt['candidate_sha256']})
    state['latest_messages_path'] = str(run_dir / 'feedback_prompt.json')
    state['status'] = ('all_trials_passed' if rate == 1.0 else
                       'physical_tested; inspect result or generate the next revision in the current session')
    dump(state_path, state)
    dump(ROOT / 'agent_runs/latest_session.json', {'session_dir': str(session_dir),
        'latest_receipt': str(run_dir / 'session_receipt.json'), 'status': state['status']})
    print(json.dumps({'session_dir': str(session_dir), 'receipt': str(run_dir / 'session_receipt.json'),
        'success_rate': rate, 'status': receipt['status'], 'actual_llm_request_sent': False}, ensure_ascii=False))
    return 0 if rate >= 0.5 else 3


def generate(cfg, task_info_path):
    key, model, base_url, public = model_settings(cfg)
    if not public['ready']:
        raise RuntimeError('Missing a normally authorized LLM API key and model. Run preflight; no request was sent.')
    if not cfg.get('setup_factory'):
        raise RuntimeError('A task setup_factory is required to keep config and outputs isolated.')
    task_info = json.loads(Path(task_info_path).read_text())
    register_task_module(cfg, task_info['task_name'])
    setup = load_factory(cfg['setup_factory'])
    # The RealMan task deliberately binds its frozen native Base_Task before
    # importing current code_gen. Importing current envs first would select a
    # different robot contract; never silently do that.
    generation, testing = official_modules(cfg)
    assert_runtime_contract(cfg)
    run_dir = new_run('official_agent')
    sys.path.insert(0, str(run_dir))
    # A previous envs_gen import must not point at a historical experiment.
    for name in list(sys.modules):
        if name == 'envs_gen' or name.startswith('envs_gen.'):
            del sys.modules[name]
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=base_url or None,
                    timeout=float(cfg.get('request_timeout_seconds', 120)), max_retries=0)
    counter = 0

    def transport(messages, **kwargs):
        nonlocal counter
        counter += 1
        # Prompts/responses contain task code only, never the API key.
        dump(run_dir / f'llm_request_{counter:02d}.json', {'model': model, 'messages': messages})
        try:
            response = client.chat.completions.create(model=model, messages=messages,
                temperature=float(cfg.get('temperature', 0)), stream=False)
        except Exception as exc:
            # Avoid serializing exception text that a provider could populate with credentials.
            dump(run_dir / f'llm_error_{counter:02d}.json', {
                'error_type': type(exc).__name__, 'status_code': getattr(exc, 'status_code', None),
                'note': 'Provider message suppressed to avoid accidental credential disclosure.'})
            raise RuntimeError('LLM request failed; see the redacted llm_error JSON.') from None
        content = response.choices[0].message.content
        if not isinstance(content, str) or 'def play_once' not in content or '```' not in content:
            raise RuntimeError('LLM returned no fenced play_once method expected by the original parser.')
        (run_dir / f'llm_response_{counter:02d}.txt').write_text(content)
        return content

    def setup_isolated(task_name):
        task, args = setup(task_name=task_name, runtime_config=cfg, run_dir=str(run_dir))
        expected_module = f'envs_gen.gpt_{task_name}'
        if task.__class__.__name__ != f'gpt_{task_name}' or task.__class__.__module__ != expected_module:
            raise RuntimeError('The setup factory did not load the actual LLM-generated expert; refusing a reference fallback.')
        return task, args

    original_run = testing.run
    generation.generate = transport
    generation.setup_task_config = setup_isolated
    generation.run = lambda task, args: original_run(task, args, check_num=int(cfg['check_num']))
    before = source_audit(cfg)
    dump(run_dir / 'run_metadata.json', {'llm': public, 'source': before,
        'task_name': task_info['task_name'], 'check_num': cfg['check_num'],
        'actual_imported_runtime_modules': runtime_module_audit(),
        'additional_available_env_functions': list(cfg.get('additional_available_env_functions', {}))})
    with cwd(run_dir):
        result = generation.main(task_info)
    dump(run_dir / 'result.json', {
        'best_success_rate': result[0], 'success_rates': result[1],
        'best_generated_code_available': result[2] is not None,
        'best_run_records': result[3], 'llm_request_count': counter,
        'actual_llm_request_sent': counter > 0, 'source_unchanged': before == source_audit(cfg),
        'note': 'Original Agent success threshold is 0.5; this does not imply dataset-collection acceptance.'})
    print(str(run_dir))
    return 0 if result[0] >= 0.5 else 3


def session_postprocess(cfg, session_dir, candidate_path):
    """Exercise the original post-response parser using a real session file.

    No new physical trials or API calls occur. Physical proof remains the
    separately linked, immutable original-tester receipt.
    """
    session_dir = Path(session_dir).resolve()
    session_dir.relative_to((ROOT / 'agent_runs').resolve())
    state = json.loads((session_dir / 'session_state.json').read_text())
    if not state.get('attempts'):
        raise RuntimeError('A completed session-test is required before linking parser equivalence.')
    physical_receipt_path = Path(state['attempts'][-1]['receipt_path']).resolve()
    physical = json.loads(physical_receipt_path.read_text())
    if physical.get('status') != 'physical_test_complete':
        raise RuntimeError('The linked original tester did not finish.')
    candidate = Path(candidate_path).resolve()
    if sha(candidate) != physical['candidate_sha256']:
        raise RuntimeError('File bridge candidate differs from the physically tested candidate.')
    task_info = state['task_info']
    task_name = task_info['task_name']
    register_task_module(cfg, task_name)
    setup = load_factory(cfg['setup_factory'])
    generation, testing = official_modules(cfg)
    assert_runtime_contract(cfg)
    before = source_audit(cfg)
    task_sources = {name: sha(ROOT / name) for name in physical['task_implementation_sha256']}
    if task_sources != physical['task_implementation_sha256']:
        raise RuntimeError('Task implementation changed since the linked physical test.')
    run_dir = new_run('session_file_bridge_postprocess')
    sys.path.insert(0, str(run_dir))
    for name in list(sys.modules):
        if name == 'envs_gen' or name.startswith('envs_gen.'):
            del sys.modules[name]
    importlib.invalidate_caches()
    response = '```python\n' + candidate.read_text() + '\n```'
    (run_dir / 'current_session_file_bridge_response.txt').write_text(response)
    details = {}

    def method_ast(path):
        tree = ast.parse(Path(path).read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == f'gpt_{task_name}']
        if len(classes) != 1:
            raise RuntimeError('Expected exactly one generated task class.')
        methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == 'play_once']
        if len(methods) != 1:
            raise RuntimeError('Expected exactly one play_once method.')
        return tree, classes[0], ast.dump(methods[0], include_attributes=False)

    def file_transport(message, **kwargs):
        dump(run_dir / 'original_generation_prompt.json', {'messages': message,
            'original_transport_kwargs': kwargs,
            'actual_transport': 'current-session real candidate file bridge',
            'actual_llm_request_sent': False})
        return response

    def isolated_setup(name):
        return setup(task_name=name, runtime_config=cfg, run_dir=str(run_dir))

    def parser_import_validation_only(task, args):
        generated = run_dir / 'envs_gen' / f'gpt_{task_name}.py'
        tree, cls, output_method = method_ast(generated)
        _, _, input_method = method_ast(candidate)
        equal = input_method == output_method
        if not equal:
            raise RuntimeError('Original-parser play_once differs from the physically tested method AST.')
        if len([n for n in tree.body if isinstance(n, ast.ClassDef)]) != 1 or not all(
                isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef)) for n in tree.body):
            raise RuntimeError('Original parser output contains unexpected top-level statements.')
        if len(cls.body) != 1 or not isinstance(cls.body[0], ast.FunctionDef):
            raise RuntimeError('Original parser output class contains more than play_once.')
        if not (len(cls.bases) == 1 and isinstance(cls.bases[0], ast.Name) and cls.bases[0].id == task_name):
            raise RuntimeError('Original parser output has a different task inheritance.')
        if task.__class__.__module__ != f'envs_gen.gpt_{task_name}' or task.__class__.__name__ != f'gpt_{task_name}':
            raise RuntimeError('The original-parser class was not actually imported by the factory.')
        details.update(generated_file=str(generated), generated_sha256=sha(generated),
            input_candidate_sha256=sha(candidate), method_ast_equal=equal,
            generated_template_structure_passed=True, generated_class_import_passed=True)
        # Deliberate parser-only boundary; never fabricate a physical score.
        return None, 'Parser/import verification only; no new physical trials.', 0, []

    generation.generate = file_transport
    generation.setup_task_config = isolated_setup
    generation.run = parser_import_validation_only
    with cwd(run_dir):
        result = generation.generate_code(task_info)
    if len(result) != 5 or result[1] is not None or not details:
        raise RuntimeError('Original generate_code did not finish the declared parser-only validation.')
    receipt = dict(details, original_generate_code_post_response_executed=True,
        generation_transport='current-session real candidate file bridge; user-authorized',
        actual_llm_request_sent=False, API_request=False,
        linked_physical_test_receipt=str(physical_receipt_path), new_physical_trials=0, score=None,
        task_implementation_sha256=task_sources, task_matches_linked_physics=True,
        source_unchanged=before == source_audit(cfg),
        note='Original parser/write/import verified. Score remains null; physical evidence is the linked original run receipt.')
    dump(run_dir / 'postprocess_receipt.json', receipt)
    dump(ROOT / 'agent_runs/latest_parser.json', {'receipt': str(run_dir / 'postprocess_receipt.json'),
        'generated_file': details['generated_file'], 'generated_sha256': details['generated_sha256']})
    print(json.dumps({'receipt': str(run_dir / 'postprocess_receipt.json'), **details}, ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'prompt', 'generate', 'session-export', 'session-test',
                                       'session-postprocess'])
    parser.add_argument('--config')
    parser.add_argument('--task-info')
    parser.add_argument('--session-dir')
    parser.add_argument('--candidate')
    parser.add_argument('--check-num', type=int)
    parser.add_argument('--output', default=str(ROOT / 'agent_preflight.json'))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.mode == 'preflight':
        return preflight(cfg, args.output)
    if args.mode in ('session-test', 'session-postprocess'):
        if not args.session_dir or not args.candidate:
            parser.error(f'{args.mode} requires --session-dir and --candidate')
        if args.mode == 'session-test':
            return session_test(cfg, args.session_dir, args.candidate, args.check_num)
        return session_postprocess(cfg, args.session_dir, args.candidate)
    if not args.task_info:
        parser.error('--task-info is required for prompt and generate')
    if args.mode in ('prompt', 'session-export'):
        return prepare_prompt(cfg, args.task_info, session_mode=args.mode == 'session-export',
                              candidate_path=args.candidate)
    return generate(cfg, args.task_info)


if __name__ == '__main__':
    raise SystemExit(main())
