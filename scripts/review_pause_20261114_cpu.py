"""Final saved-data analysis only. No model construction, forward, or update."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import shutil
import torch

P = Path(__file__).resolve().parents[1]
W = P.parent
OUT = W / 'outputs/FDT_CURRENT_COMPLETE_ANALYSIS_20261114_evidence'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def save(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def independent_decode(trace):
    ranges = ((128, 191), (160, 191), (144, 191), (128, 159), (128, 143))
    choices = trace['choices'].tolist()
    total = trace['patch_count'] * 16
    out, pending, low, high, eos = [], 0, 128, 191, False
    assert len(choices) == 9 and all(len(x) == total for x in choices)
    for position in range(total):
        state = 4 + ranges.index((low, high)) if pending else min(total-position, 4)-1
        b = choices[state][position]
        if pending:
            assert low <= b <= high
            pending -= 1
        elif b == 256:
            eos = True
            break
        elif b < 128:
            pending = 0
        elif 194 <= b <= 223 and total-position >= 2:
            pending = 1
        elif 224 <= b <= 239 and total-position >= 3:
            pending = 2
        elif 240 <= b <= 244 and total-position >= 4:
            pending = 3
        else:
            raise AssertionError('illegal UTF8 decision')
        out.append(b)
        low = 160 if b == 224 else 144 if b == 240 else 128
        high = 159 if b == 237 else 143 if b == 244 else 191
    assert pending == 0
    return bytes(out).decode('utf-8'), eos, len(out)


def validate_trace(trace, expected):
    labels = torch.full((trace['patch_count']*16,), 257, dtype=torch.long)
    raw = expected.encode('utf-8'); n = len(raw)
    labels[:n] = torch.tensor(list(raw)); labels[n] = 256
    a = torch.arange(258)
    masks = []
    for room in (1, 2, 3, 4):
        masks.append((a < 128) | (a == 256) | ((a >= 194) & (a <= 223) & (room >= 2))
                     | ((a >= 224) & (a <= 239) & (room >= 3))
                     | ((a >= 240) & (a <= 244) & (room >= 4)))
    masks += [(a >= lo) & (a <= hi) for lo, hi in ((128,191),(160,191),(144,191),(128,159),(128,143))]
    for patch, z in zip(trace['sample_patch_indices'].tolist(), trace['sample_full_logits']):
        start = patch * 16
        assert torch.equal(z.argmax(-1).to(torch.int16), trace['argmax'][start:start+16])
        choices = torch.stack([z.masked_fill(~m, -torch.inf).argmax(-1).to(torch.int16) for m in masks])
        assert torch.equal(choices, trace['choices'][:,start:start+16])
        lp = z.gather(-1, labels[start:start+16,None]).squeeze(-1) - z.logsumexp(-1)
        assert torch.allclose(lp, trace['gold_log_probability'][start:start+16], atol=3e-5, rtol=1e-5)
    return dict(NLL=float(-trace['gold_log_probability'][:n+1].mean()),
                byte_CE=float(-trace['gold_log_probability'][:n].mean()),
                EOS_CE=float(-trace['gold_log_probability'][n]),
                argmax_byte_accuracy=float((trace['argmax'][:n].long()==labels[:n]).float().mean()))


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    run = P / 'runs/grid_tail_thought_20261114_v105'
    result, old_review = read(run/'result.json'), read(run/'root_saved_review.json')
    terminal = read(run/'ROOT_terminal_native.json')
    assert terminal['Python_count'] == 0 and terminal['all_owned_native_proxy_gone'] and terminal['before_analysis']
    selection = read(P/'configs/frozen_thought_codec_20261113_v94.json')
    refs = [(selection['selected_runtime'], selection['selected_SHA']),
            (selection['mandatory_standalone_base'], selection['mandatory_base_SHA'])]
    refs += [(x['path'], x['SHA']) for x in result['checkpoints']]
    identities = []
    for path, expected in refs:
        actual = sha(path); assert actual == expected
        identities.append(dict(path=path, bytes=Path(path).stat().st_size, SHA=actual))
    save('checkpoint_identities.json', identities)
    data = P / 'research/artifacts/memory_thought_data_20261113_v95_cpu_v1'
    features, targets = {}, {}
    split_statistics = {}
    for split in ('train','dev_pair','dev_surface','dev_position'):
        ff, tt = read(data/(split+'_features.json')), read(data/(split+'_targets.json'))
        assert all(set(f)=={'id','prompt'} for f in ff)
        assert {f['id'] for f in ff} == {t['id'] for t in tt}
        features.update({f['id']: f['prompt'] for f in ff}); targets.update({t['id']: t for t in tt})
        split_statistics[split] = dict(rows=len(tt), unique_answer_strings=len({t['answer'] for t in tt}),
            labels=dict(collections.Counter(t['label'] for t in tt)), max_input_UTF8bytes=max(len(f['prompt'].encode()) for f in ff))
    partial_unknown = {
        'e900c6d27a2565b8e2454c182f1a1fb4c478c87117a53a0ed920ac74986bd313',
        'c68af07b382474ad260174735a2995891b39deae37a37ac8261379d025ac6200',
        '3cf2c1490448867c5578b51f12b7a5062094d0750e1dd74474a72bc38671963d',
        '92f5cf15dad89a5c3abe787157aa1f8671b609aa1ba867c4159f967a794dbfcc',
        'a4b0d476cf44607ac66539a2162fb320c4fc82e2bdf57dc655a45b1db11fd530',
        '700ba39a1ea6a95abeece6269633a3f7a4b3cafc914c83fcf8ddf84dc245739b',
        '5e46af4c621f7d4204d3e66a41b45916681d3aecfc828800fde49871380ade18',
        '46f4d990538da4ebc63e01f70b93b326ea26a31bd1e4b62e098fe2a43588944d'}
    normal = [x for x in read(run/'assessment.json') if x['phase']=='final']
    assert len(normal) == 96
    reviewed = []
    for row in normal:
        assert row['expected'] == targets[row['input_id']]['answer']
        assert sha(row['raw']['path']) == row['raw']['SHA']
        raw = torch.load(row['raw']['path'], map_location='cpu', weights_only=True)
        text, eos, length = independent_decode(raw['trace'])
        assert text == row['output'] and eos == row['EOS']
        scores = validate_trace(raw['trace'], row['expected'])
        assert abs(scores['NLL']-row['byte_NLL']) < 1e-6
        category = 'UNCERTIFIED_garbled_or_incomplete_not_a_semantic_zero_claim'
        if row['input_id'] == 'f1e7c9858229505b7ce905742238dda767e6f4f0d4af6de1d6c7cf5b078195eb':
            category = 'correct_unknown_cue_with_corrupt_prefix_not_clean_sentence'
        elif row['input_id'] in partial_unknown:
            category = 'possible_unknown_intent_misspelled_NOT_certified'
        elif row['input_id'] == '7c5053d93b97814548163b8b5317c15b66a153018646076c63ddf81b84edf772':
            category = 'wrong_direction_if_bare_operator_is_relation_of_first_to_second'
        reviewed.append(dict(input_id=row['input_id'],split=row['split'],label=row['label'],
            prompt=features[row['input_id']],expected=row['expected'],output=text,manual_category=category,
            scores=scores,output_bytes=length,early_EOS=bool(eos and length<len(row['expected'].encode()))))
    save('V105_all96_source_output_review.json', reviewed)
    counts = dict(collections.Counter(r['manual_category'] for r in reviewed))
    md = ['# V105 전체96개 입력·출력 판독 기록', '',
        '실제 질문을 기준으로 판독했다. 문자열 불일치를 곧바로 의미 오류로 세지 않고, 불명확한 철자도 인증된 정답으로 올리지 않는다. 독립 블라인드 의미 정확도 표가 아니다.', '',
        '|번호/패널|입력|기대 답변|실제 출력(JSON 이스케이프)|판독|','|---|---|---|---|---|']
    def cell(s):
        return s.replace('|','&#124;').replace('\n','\\n')
    for i,r in enumerate(reviewed):
        md.append('|'+ '|'.join(cell(str(v)) for v in (str(i+1)+'/'+r['split'],r['prompt'],r['expected'],json.dumps(r['output'],ensure_ascii=False),r['manual_category']))+'|')
    # Korean report is the primary explanation; this table is a verbatim source/output appendix.
    with (OUT/'V105_all96_source_output_review.md').open('x',encoding='utf-8',newline='\n') as f:
        f.write('\n'.join(md)+'\n')
    diag = P / 'research/artifacts/thought_rollout_20261114_v106_cpu_v1'
    dr = read(diag/'cpu_preflight_owned/runtime/result.json')
    dt, ds = read(diag/'ROOT_terminal_native.json'), read(diag/'cpu_preflight_owned/final_status.json')
    assert dt['Python_count']==0 and dt['before_analysis'] and dt['all_CPU_diagnostic_processes_gone']
    assert ds['exit']['all_owned_processes_gone'] and ds['worker_exit_code']==0 and not ds['owner_imported_Torch']
    raw_path = diag/'cpu_preflight_owned/runtime/raw.pt'
    assert sha(raw_path)==dr['raw_SHA']
    raw_rows = torch.load(raw_path,map_location='cpu',weights_only=True)
    assert len(raw_rows)==len(dr['records'])==12
    diag_review = []
    for saved,row in zip(raw_rows,dr['records']):
        assert saved['id']==row['id'] and saved['steps']==row['steps']
        text,eos,length=independent_decode(saved['trace']);scores=validate_trace(saved['trace'],row['expected'])
        assert text==row['output'] and eos==row['EOS']
        assert abs(scores['byte_CE']-row['byte_CE'])<1e-6 and abs(scores['EOS_CE']-row['EOS_CE'])<1e-6
        diag_review.append(dict(row,prompt=features[row['id']],scores=scores,output_bytes=length))
    grouped = collections.defaultdict(list)
    for r in diag_review:grouped[r['id']].append(r)
    assert len(grouped)==4 and all([x['steps'] for x in rr]==[1,4,16] and len({x['noise_seed'] for x in rr})==1 for rr in grouped.values())
    save('V106_independent_saved_review.json',dict(raw_SHA=dr['raw_SHA'],records=diag_review,
        all12_saved_decodes_CE_sampled_full258_verified=True,all4_same_source_seed_across_step_counts=True,
        known_correct_unknown_1step2_then_16step0=True,positive_correct0_at_all_counts=True,
        no_extra_NN_calls=True,no_model_construction=True,NOT_all258_all_positions=True))
    stats = old_review['group_summaries'];metric_table = []
    first = read(P/'runs/memory_thought_20261114_v102/root_saved_review.json')['group_summaries']
    for split in ('dev_pair','dev_surface','dev_position'):
        metric_table.append(dict(split=split,
            random_thought_NLL=first['baseline/'+split]['mean_NLL'],
            V102_NLL=stats['baseline_inherited_V102/'+split]['mean_NLL'],
            V105_NLL=stats['final/'+split]['mean_NLL'],
            V105_minus_V102=stats['final/'+split]['mean_NLL']-stats['baseline_inherited_V102/'+split]['mean_NLL'],
            V102_byte_argmax_accuracy=stats['baseline_inherited_V102/'+split]['mean_byte_argmax_accuracy'],
            V105_byte_argmax_accuracy=stats['final/'+split]['mean_byte_argmax_accuracy']))
    info = dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        status='PAUSED_BY_HUMAN_TIMER_DELETED_ANALYSIS_ONLY',automation_id='fdt',automation_delete_confirmed=True,
        new_training_attempts=0,new_NN_forward_calls=0,new_optimizer_steps=0,GPU_calls=0,
        checkpoint_identities_verified=identities,latest_trained='V105',latest_diagnostic='V106_closed',
        V107='CPU_candidate_prior4tests_pass_no_fullsize_trial_no_runner_plan_seal_launch',
        selected_codec_parameters=123497080,thought_parameters=3224322,total_stored_parameters=126721402,
        thought_updates_V102_plus_V105=960,unique_TRAIN_rows384_not7680_independent_rows=True,
        cumulative_TRAIN_row_selections7680=True,target_UTF8_exposure_bytes135680=True,
        train_unique_answer_strings=27,all_train_dev_unique_answers=31,dataset=split_statistics,
        normal96_semantic_review_categories=counts,positive_rows48_no_unambiguous_correct_relation=True,
        semantic_accuracy_not_identical_to_exact_rate=True,metric_table=metric_table,
        mean_NLL_V102=sum(x['V102_NLL'] for x in metric_table)/3,
        mean_NLL_V105=sum(x['V105_NLL'] for x in metric_table)/3,
        final_EOS96_early72_same_length17_late7=True,
        long_codec_byte_error_rate_percent=49/524071*100,
        numeric_stress_clear_wrong_rate_percent=4/64*100,
        codec_local_FP32_storage_ratio_per16bytes=512*4/16,
        exact0_does_not_imply_semantic0=True,
        K_P='learned canonical anchors/prior active; certified factual writer absent',
        K_C='input-dependent long/recent fuzzy snapshots active; no cross-document certified consolidation',
        K_E='invalid NULL; no content/copy; internal explicit-copy policy not implemented',
        all_old_budgets_closed=True,no_model_or_registry_selection_changed=True,
        current_Cfree_bytes=shutil.disk_usage(W).free,CUDA_initialized=torch.cuda.is_initialized())
    assert not info['CUDA_initialized']
    save('analysis_summary.json',info)
    pins = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file(): pins[path.name]=dict(bytes=path.stat().st_size,SHA=sha(path))
    for relative in ['research/runtime_aligned_thought_v107.py','tests/test_runtime_aligned_thought_v107.py',
                     'research/memory_thought_diffusion_v94.py','research/fuzzy_roundtrip_v64.py',
                     'research/ordered_fuzzy_readout_v55.py','research/commit_patch_readout_v54.py',
                     'research/thought_grid_tail_v104.py','scripts/review_pause_20261114_cpu.py']:
        pins[str(P/relative)]=dict(bytes=(P/relative).stat().st_size,SHA=sha(P/relative))
    save('evidence_SHA.json',pins)
    print(json.dumps(dict(status=info['status'],metrics=metric_table,manual_categories=counts,
                         CfreeGiB=info['current_Cfree_bytes']/1024**3,NNcalls0=True),ensure_ascii=False))


if __name__=='__main__':
    main()
