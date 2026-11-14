"""Create the new saved-only reviewer, including ROOT's actual V30 text reading."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'scripts/review_owt_state_reconstruction_v29_cpu.py').read_text(encoding='utf-8')
dev=[
"He didn't is retained, but acknowledgement of personal culpability and deceit against the American public are absent.",
"Fentages is not Fans; waving, flowers, anticipation and standing absent.",
"Uppercase and not know fragments do not preserve Disney, donut sundae or the source surprised question.",
"Globally is exactly recovered, but few ransom payments and victim backup restoration absent; unrelated claims substituted.",
"Dethat sporting is not Deathstroke; exact panel, team, Slade Wilson and Moscow absent.",
"War-like opening is malformed; Israel, Syria, Assad and stable ceasefire/governance are absent.",
"She fared is not failed; biology replaces sobriety, DUI charge absent.",
"I and pay-like fragments do not preserve taking goods/leaving Yanick's stall without paying.",
"But more into promise is not the source proposition; no explosion and effective workplace speech restriction absent.",
"Name-like fragments and added not do not preserve Fabbri, Chiarelli, Bowman, barely on radar or three months.",
"We survives but asking second academic, review and anonymous article absent.",
"Weubly is not Wubbels; lawyer, inaction and primary concerns absent.",
"Asian/name-like fragments and Dermany are not the six-country vessel-holding event.",
"CBS/name-like text replaces Jamaican bobsled qualification for Sochi Olympics.",
"Japanese is not Janissa; preventing assassination with wall absent.",
"unfinget/string-like spelling is insufficient; named functions and temporary storage/pointer array relationship absent."
]
train=[
"More people, some in critical condition is exactly recovered: the partial descriptive proposition that some people are critical is retained. Arriving late Friday is absent and later text malformed. Partial condition only, not whole faithful reconstruction.",
"They also depended replaces delay postponed; Clinton, five primaries, definitive answer and Tuesday uncertainty absent.",
"Outside the court is recovered, but Brazil is not Brazau; future appeal of sentence absent.",
"Israndy is not Eduardo; cash-payment request and bank/card fee avoidance absent.",
"NATO is exactly recovered again; operators, thumb-drive purchase/use and Russian infiltration chain absent. Entity only.",
"Problem/will have does not preserve probable opinions about bike; qualifiers and target absent.",
"The tweets remains but flagged, source and forwarding to narrator absent; previous were opening lost.",
"Once the first season is remains; ending and no-more qualifying tournaments absent.",
"Bill Haslam is exactly recovered; taking part in panel discussion with Vermont governor absent. Entity only; source selected fragment.",
"Donations and correspondence remains; support, sending to family and exact postal address absent;10/BBC invented.",
"It just goes to show you the exact is retained, but more croplians does not express moral compass; no preserved moral-assessment proposition.",
"Only U.S. reference is exact; forces, diplomats in Iraq and reaffirming absent. Input fragment receives no invented complete gold.",
"I'm good is not reliably the same agreement referent; My Other Home, screening, China and August11 lost,1966 invented.",
"We'll be and2012 are fragments; taking orders and exact interval/shipment schedule absent,2015 invented.",
"Craig Charles remains exact; other three participants and joining relation absent. Entity only.",
"She then told the remains but25 becomes20; cop identity and not-to-move instruction absent."
]
start=s.index('DEV_COMMENTS=');end=s.index('\ndef read(',start)
s=s[:start]+'DEV_COMMENTS='+repr(dev)+'\nTRAIN_COMMENTS='+repr(train)+'\n'+s[end:]
s=s.replace('reconstruction_20261029_v29','reconstruction_20261029_v30').replace('V29_STATE_RECONSTRUCTION','V30_STATE_RECONSTRUCTION').replace('V29 same-input','V30 same-input')
s=s.replace("==25924","==30164").replace("==31616","==43872")
start=s.index("    assert initial['optimizer_parameter_names'][:2]");end=s.index('    full=[]',start)
s=s[:start]+'''    components=('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
    source_components=support.state_sha({k:parent[k] for k in components})
    assert source_components==support.state_sha({k:initial[k] for k in components})
    for k in ('history','orders','exposure','evaluations','ancestry','actual_calls'):assert support.state_sha(initial['ancestry'][k])==support.state_sha(parent[k])
    assert initial['migration']==parent['migration']==initial['ancestry']['source_first_thaw_migration']
    assert initial['optimizer_parameter_names']==parent['optimizer_parameter_names']
    a,b=adam_by_name(parent),adam_by_name(initial)
    assert len(a)==len(b)==81 and all(support.state_sha(a[n])==support.state_sha(b[n]) for n in a)
    old_names=set(n for g in initial['optimizer_parameter_names'][:2] for n in g)
    added=initial['optimizer_parameter_names'][2];phase_names=set(b)
    assert len(old_names)==63 and len(added)==18
    buffers={'core.basis','core.frequencies'}
    assert support.state_sha({n:v for n,v in initial['model'].items() if n not in phase_names|buffers})==initial['frozen_sha']==parent['frozen_sha']
    source_scaler=dict(parent['scaler']);del parent,a,b
'''+s[end:]
s=s.replace("(2048,run/'model/checkpoints/state_2048.pt'),(4096,run/'model/checkpoints/state_4096.pt')","(4096,run/'model/checkpoints/state_4096.pt'),(8192,run/'model/checkpoints/state_8192.pt')")
s=s.replace('fdt_owt_fuzzy_AE_v29_fullstate','fdt_owt_fuzzy_AE_v30_fullstate')
s=s.replace("==88005+s['updates']","==92099+s['updates']").replace("(states[n] is None if local==0 else int(states[n]['step'])==s['updates'])","int(states[n]['step'])==4094+s['updates']")
s=s.replace("'old_Adam_step':88005+s['updates'],'new_Adam_step':s['updates'] if local else None","'old_Adam_step':92099+s['updates'],'new_Adam_step':4094+s['updates']")
s=s.replace('if local==4096:state=s','if local==8192:state=s').replace('(4096,4094,32768,2)','(8192,8188,65536,4)').replace("==92132","==100324").replace("==92099","==100287")
# The source old-state base is92099, not the final global count.
s=s.replace("==100287+s['updates']","==92099+s['updates']")
s=s.replace("    assert sorted(i for b in state['orders'] for i in b)==list(range(32768))","    for offset in (0,4096):assert sorted(i for b in state['orders'][offset:offset+4096] for i in b)==list(range(32768))")
s=s.replace("==2926478 and r['exposure']['selected_rows']==32768","==5852956 and r['exposure']['selected_rows']==65536")
s=s.replace('history[i:i+2048]','history[i:i+4096]').replace('for i in (0,2048)','for i in (0,4096)')
s=s.replace("'encode':4096+32+4+4","'encode':8192+32+4+4").replace("'decoder':8192+96","'decoder':16384+96")
s=s.replace("'new_attempts':4096,'new_updates':4094,'global_attempts':92132,'global_updates':92099","'new_attempts':8192,'new_updates':8188,'global_attempts':100324,'global_updates':100287")
s=s.replace("'source_old63Adam_model_scaler_allRNG_exact':True,'source_whole_optimizer_structure_unchanged':False,'new18_initial_Adam_empty':True","'source_ALL81Adam_model_scaler_allRNG_exact':True,'source_whole_optimizer_structure_unchanged':True,'new18_initial_Adam_empty':False")
s=s.replace("'online_half_epoch_means':online","'online_epoch_means':online").replace("'online_half_epoch_means','total_new_calls'","'online_epoch_means','total_new_calls'")
s=s.replace("'ROOT_partial_proposition':0,'ROOT_nonblind'","'ROOT_partial_proposition':1,'ROOT_nonblind'")
s=s.replace("                if panel=='train' and i==14:item.update(meaning_grade='entity_fragment_only',preserved_named_entities=['Craig Charles'])", "                if panel=='train' and i in (4,8,14):item.update(meaning_grade='entity_fragment_only',preserved_named_entities=[{4:'NATO',8:'Bill Haslam',14:'Craig Charles'}[i]])\n                if panel=='train' and i==0:item.update(meaning_grade='partial_condition_proposition',partial_proposition=True,preserved_partial_proposition='Some of the people are in critical condition; arrival/time not preserved.')")
s=s.replace("summary.update(ROOT_read_all16=True,ROOT_faithful_paraphrase=0,ROOT_partial_proposition=0)","summary.update(ROOT_read_all16=True,ROOT_faithful_paraphrase=0,ROOT_partial_proposition=int(mode=='train_normal'))")
s=s.replace('An initial one-epoch thaw does not establish whole FDT/fuzzy failure or completion.','Three total epochs after thaw do not establish whole FDT/fuzzy failure or completion; whole reconstruction still unestablished.')
with (R/'scripts/review_owt_state_reconstruction_v30_cpu.py').open('x',encoding='utf-8') as f:f.write(s)
