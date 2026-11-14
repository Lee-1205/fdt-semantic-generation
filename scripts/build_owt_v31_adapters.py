"""New import schema/owner; numerical model and optimizer contract stay unchanged."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
e=(R/'scripts/owt_fuzzy_engine_v30.py').read_text(encoding='utf-8')
e=e.replace("if s['schema']!='fdt_owt_fuzzy_AE_v29_fullstate'","if s['schema']!='fdt_owt_fuzzy_AE_v30_fullstate'").replace("actual V29 source","actual V30 source")
e=e.replace("s['schema']='fdt_owt_fuzzy_AE_v30_fullstate';return s","s['schema']='fdt_owt_fuzzy_AE_v31_fullstate';s['worker_stage']=getattr(self,'worker_stage','new_allocation');return s")
e=e.replace("if s['schema']!='fdt_owt_fuzzy_AE_v30_fullstate':raise ValueError('V30 whole driver schema')","if s['schema']!='fdt_owt_fuzzy_AE_v31_fullstate':raise ValueError('V31 whole driver schema')")
e=e.replace("adapted['schema']='fdt_owt_fuzzy_AE_v29_fullstate';super().restore(adapted)","adapted['schema']='fdt_owt_fuzzy_AE_v29_fullstate';super().restore(adapted);self.worker_stage=s['worker_stage']")
with (R/'scripts/owt_fuzzy_engine_v31.py').open('x',encoding='utf-8') as f:f.write(e)
o=(R/'scripts/supervise_owt_fuzzy_reconstruction_20261029_v30.py').read_text(encoding='utf-8')
o=o.replace('reconstruction_v30','reconstruction_v31').replace('RECONSTRUCTION_V30','RECONSTRUCTION_V31')
o=o.replace("'attempts':8192","'attempts':2048").replace("'max_calls':28000","'max_calls':12000").replace("plan.get('epochs')!=2","plan.get('epochs')!=256").replace('!=26776','!=11016')
with (R/'scripts/supervise_owt_fuzzy_reconstruction_20261029_v31.py').open('x',encoding='utf-8') as f:f.write(o)
