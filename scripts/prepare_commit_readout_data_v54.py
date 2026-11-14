"""512 hash-selected existing TRAIN documents; same16 observed DEV inputs."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    output=ROOT/'research/artifacts/commit_readout_20261111_v54_cpu'
    if output.exists():raise FileExistsError('immutable artifact')
    old=ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu'
    train=json.loads((old/'train.json').read_text());dev=json.loads((old/'dev.json').read_text())
    train=sorted(train,key=lambda row:hashlib.sha256(('v54:'+row['id']).encode()).hexdigest())[:512]
    active=sum(math.ceil((row['actual_bytes']+1)/16) for row in train)
    cached=sum(math.ceil(row['actual_bytes']/16)+1 for row in train)
    attempts=6*math.ceil(active/256)
    cost=dict(encode=528,train_readout=attempts,final3condition_readout=3*math.ceil(sum(math.ceil(row['actual_bytes']/16)+1 for row in dev)/256))
    cost['total']=sum(cost.values())
    manifest=dict(train_documents=512,dev_documents=16,train_UTF8_bytes=sum(row['actual_bytes'] for row in train),
        train_active_patches=active,train_cached_patches=cached,patch_bytes=16,epochs=6,batch=256,attempts=attempts,
        estimated_feature_target_bytes=cached*(512*4+16*8+16)+512*2*512*4,cost=cost,
        source=str(old),new_downloads=0,held_training=0,E2_access=0,
        purpose='freeze V53 input encoder and all3 memories; train only shared Commit-to-bytes probe')
    output.mkdir()
    for name,value in [('train',train),('dev',dev),('manifest',manifest)]:
        (output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest))
if __name__=='__main__':main()
