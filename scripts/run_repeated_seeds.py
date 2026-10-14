"""Execute the registered two additional paired MSVD seeds."""
from __future__ import annotations
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
import torch
from scripts.run_full_msvd_infonce_ablation import (
    A_ROOT, SOURCE_FILES as BASE_SOURCES, verify_registered_baseline,
    cached_caption_infonce_objective, retrieval_diagnostic, primary_decision,
    assert_trace_equal, load_strict,
)
from scripts.run_full_msvd_balanced import TRAIN, VAL, DATA_SHA256, evaluate_validation, trace_evidence
from scripts.run_caption_policy_ab import (
    cached_caption_loss, tokenize_caption, tokenize_references, reference_seed,
    raw_backbone_features, write_json,
)
from scripts.run_overfit16 import component_hashes, configure_trainable, tensor_sha
from scripts.eval_retrieval import encode_texts
from scripts.diagnose_prefix_sensitivity import donor_permutation
from src.cli.train_caption_mapper import set_seed
from src.models.caption_model import VideoCaptionModel
from src.data.data_loader import build_dataloader
from src.training.checkpoint import sha256_file, save_training_checkpoint, capture_rng_state, restore_rng_state

SEEDS = (124, 125)
SOURCES = tuple(dict.fromkeys((*BASE_SOURCES, 'scripts/run_repeated_seeds.py',
    'experiments/repeated_seeds_protocol_20260908.md', 'src/cli/train_caption_mapper.py')))

def make_schedule(records, seed):
    if seed not in SEEDS:
        raise ValueError('Only registered seeds 124 and 125 are allowed')
    generator = torch.Generator().manual_seed(seed + 333)
    batches = [torch.randperm(len(records), generator=generator).reshape(-1, 8).tolist()
               for _ in range(66)]
    orders = {str(r['video_id']): torch.randperm(len(r['captions']),
        generator=torch.Generator().manual_seed(reference_seed(str(r['video_id'])) + seed - 123)).tolist()
        for r in records}
    return {'seed': seed, 'batch_seed': seed + 333, 'batches': batches, 'orders': orders}

def objective(model, features, captions, arm):
    if arm == 'B':
        return cached_caption_infonce_objective(model, features, captions)
    loss = cached_caption_loss(model, features, captions)
    return {'loss': loss, 'caption_loss': loss, 'alignment_loss': loss.new_zeros(())}

def smoke_gradient_check(model, features, captions, seed):
    model.train()
    model.encoder.backbone.eval()
    set_seed(seed)
    before = capture_rng_state()
    a = cached_caption_loss(model, features, captions)
    after_a = capture_rng_state()
    restore_rng_state(before)
    b = cached_caption_infonce_objective(model, features, captions)
    after_b = capture_rng_state()
    if not torch.equal(a, b['caption_loss']):
        raise RuntimeError('A/B initial caption CE differs')
    if not torch.equal(after_a['torch_cpu'], after_b['torch_cpu']) or any(
        not torch.equal(x, y) for x,y in zip(after_a['torch_cuda'], after_b['torch_cuda'])):
        raise RuntimeError('InfoNCE changes RNG consumption')
    model.zero_grad(set_to_none=True)
    b['alignment_loss'].backward()
    for module in (model.encoder.proj, model.decoder.mapper):
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters()):
            raise RuntimeError('Alignment gradient does not reach a trainable module')
    model.zero_grad(set_to_none=True)
    return {'initial_ce_equal': True, 'rng_consumption_equal': True, 'alignment_gradient_pass': True}

def run_arm(model, arm, seed, folder, initial_path, initial_hashes, schedule,
            records, features, val_records, val_features, refs, texts, gold,
            token_cache, source_hashes, baseline_pre, smoke):
    folder.mkdir()
    load_strict(model, initial_path)
    assert component_hashes(model) == initial_hashes
    params = configure_trainable(model)
    optimizer = torch.optim.AdamW([p for _,p in params], lr=3e-4, weight_decay=0.01)
    epochs = [schedule['batches'][0][:2]] if smoke else schedule['batches']
    milestones = {1} if smoke else {5,20,40,66}
    expected_steps = 2 if smoke else 9900
    ids = [str(r['video_id']) for r in records]
    visits = Counter()
    coverage = {v: Counter() for v in ids}
    evaluations = []
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    step = 0
    donor = donor_permutation(len(val_records), seed=321)
    compare = (folder.parent / 'A/trace.jsonl').open(encoding='utf-8') if arm == 'B' else nullcontext(None)
    with compare as a_trace, (folder/'trace.jsonl').open('x',encoding='utf-8') as trace, \
         (folder/'train_metrics.jsonl').open('x',encoding='utf-8') as log:
        for epoch, batches in enumerate(epochs, 1):
            losses = []
            for indices in batches:
                step += 1
                targets = []
                for i in indices:
                    v = ids[i]
                    order = schedule['orders'][v]
                    ci = order[(epoch-1) % len(order)]
                    targets.append(records[i]['captions'][ci])
                    visits[v] += 1
                    coverage[v][ci] += 1
                batch_features = features[indices]
                captions = torch.stack([token_cache[t] for t in targets])
                evidence = trace_evidence(step, epoch-1, indices, ids, batch_features, captions, targets)
                if a_trace is not None:
                    assert_trace_equal(evidence, json.loads(a_trace.readline()))
                trace.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True)+'\n')
                trace.flush()
                model.train()
                model.encoder.backbone.eval()
                optimizer.zero_grad(set_to_none=True)
                out = objective(model, batch_features.cuda(), captions.cuda(), arm)
                if not all(torch.isfinite(out[k]) for k in ('loss','caption_loss','alignment_loss')):
                    raise RuntimeError(f'Nonfinite objective seed={seed} arm={arm} step={step}')
                out['loss'].backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for _,p in params):
                    raise RuntimeError('Missing or nonfinite gradient')
                optimizer.step()
                row = {k: float(v.detach()) for k,v in out.items()}
                row.update(step=step, epoch=epoch)
                log.write(json.dumps(row)+'\n')
                log.flush()
                losses.append(row['caption_loss'])
            print(f'[seed={seed} {arm}] epoch={epoch}/{len(epochs)} step={step} CE={sum(losses)/len(losses):.6f}',flush=True)
            if epoch in milestones:
                caption = evaluate_validation(model,val_features,val_records,refs,donor,epoch,step,'cuda')
                retrieval = retrieval_diagnostic(model,val_features,texts,gold,'cuda')
                path = folder / f'checkpoint_epoch{epoch:03d}.pt'
                contract = dict(objective_version=3, caption_objective='causal_lm_cross_entropy',
                    caption_policy='balanced_all_references', alignment_objective='infonce' if arm=='B' else 'none',
                    alignment_weight=0.1 if arm=='B' else 0.0, info_nce_temperature=0.07,
                    false_negative_policy='exact_caption', physical_batch_size=8,
                    video_representation='mean_visual_prefix_tokens_l2_normalized',
                    text_representation='frozen_gpt2_wte_masked_mean_l2_normalized',
                    frozen_gpt2=True, frozen_vit_backbone=True, cached_frozen_backbone=True)
                save_training_checkpoint(path,model=model,optimizer=optimizer,scheduler=None,
                    progress=dict(step=step,epoch=epoch,epoch_complete=True,best_val=caption['metrics']['native_nll']),
                    args=dict(gpt2_name=baseline_pre['model_source']['path'],vit_name='vit_base_patch16_224',
                        cond_mode='prefix',prefix_len=4,seed=seed,batch_size=8,max_len=64,lr=3e-4,epochs=len(epochs)),
                    training_contract=contract,dataset_contract=baseline_pre['dataset_contract'],
                    loader_state={'schedule_sha256':sha256_file(folder.parent/'schedule.json'),'num_workers':0})
                item = dict(epoch=epoch,step=step,caption=caption,retrieval=retrieval,
                    checkpoint=str(path.resolve()),checkpoint_sha256=sha256_file(path))
                evaluations.append(item)
                write_json(folder/f'evaluation_epoch{epoch:03d}.json', item)
                print(json.dumps({'seed':seed,'arm':arm,'epoch':epoch,'nll':caption['metrics']['native_nll'],
                    'gate':caption['metrics']['primary_viability_gate']['pass'],
                    't2v_r1':retrieval['true_labels']['text_to_video']['r_at_1']}),flush=True)
        if a_trace is not None and a_trace.readline():
            raise RuntimeError('Paired trace has additional steps')
    assert step == expected_steps
    if not smoke:
        assert len(visits)==1200 and set(visits.values())=={66}
        assert all(len(coverage[v])==len(records[i]['captions']) for i,v in enumerate(ids))
    final_hashes = component_hashes(model)
    assert all(final_hashes[k]==initial_hashes[k] for k in ('decoder','backbone'))
    assert all(final_hashes[k]!=initial_hashes[k] for k in ('projection','mapper'))
    load_strict(model, Path(evaluations[-1]['checkpoint']))
    assert component_hashes(model)==final_hashes
    assert retrieval_diagnostic(model,val_features,texts,gold,'cuda')==evaluations[-1]['retrieval']
    assert source_hashes=={n:sha256_file(ROOT/n) for n in SOURCES}
    assert all(sha256_file(p)==DATA_SHA256[s] for s,p in (('train',TRAIN),('val',VAL)))
    result = dict(seed=seed,arm=arm,smoke_test=smoke,paper_ready=False,test_split_read=False,
        steps=step,initial_component_hashes=initial_hashes,final_component_hashes=final_hashes,
        evaluations=evaluations,trace_sha256=sha256_file(folder/'trace.jsonl'),
        final_checkpoint_strict_reload=True,frozen_weights_unchanged=True,
        source_and_data_unchanged=True,video_visits=dict(visits),
        unique_reference_indices=sum(len(c) for c in coverage.values()),
        elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    write_json(folder/'result.json',result)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('Refusing to overwrite output directory')
    assert torch.cuda.is_available()
    base,pre=verify_registered_baseline()
    sources={n:sha256_file(ROOT/n) for n in SOURCES}
    assert all(sha256_file(p)==DATA_SHA256[s] for s,p in (('train',TRAIN),('val',VAL)))
    records=json.loads(TRAIN.read_text(encoding='utf-8'))
    validation=json.loads(VAL.read_text(encoding='utf-8'))
    train_cache=torch.load(A_ROOT/'train_features.pt',weights_only=True,map_location='cpu')
    val_cache=torch.load(A_ROOT/'val_features.pt',weights_only=True,map_location='cpu')
    assert train_cache['video_ids']==[r['video_id'] for r in records]
    assert val_cache['video_ids']==[r['video_id'] for r in validation]
    args.output_root.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    pairs=[]
    for seed in SEEDS:
        folder=args.output_root/f'seed{seed}'
        folder.mkdir()
        set_seed(seed)
        model=VideoCaptionModel(gpt2_name=pre['model_source']['path'],freeze_vit=True,vit_enable_torch_compile=False)
        initial=component_hashes(model)
        assert all(initial[k]==base['initial_component_hashes'][k] for k in ('decoder','backbone'))
        for old in [base['initial_component_hashes']]+[p['initial_hashes'] for p in pairs]:
            assert all(initial[k]!=old[k] for k in ('projection','mapper'))
        initial_path=folder/'initial_model.pt'
        torch.save({'model_state':model.state_dict()},initial_path)
        schedule=make_schedule(records,seed)
        write_json(folder/'schedule.json',schedule)
        model.cuda().eval()
        loader=build_dataloader(str(TRAIN),model.decoder.tokenizer,batch_size=8,max_len=64,
            num_frame=8,image_size=224,shuffle=False,num_wokers=0,random_caption=False)
        batch=next(iter(loader))
        with torch.inference_mode():
            raw=raw_backbone_features(model.encoder,batch['video'].cuda())
            error=float((raw.cpu()-train_cache['features'][:8]).abs().max())
        assert error<=1e-6
        configure_trainable(model)
        tokens={t:tokenize_caption(model.decoder.tokenizer,t) for r in records for t in r['captions']}
        vr=validation[:8] if args.smoke else validation
        vf=val_cache['features'][:len(vr)]
        refs={r['video_id']:tokenize_references(model.decoder.tokenizer,r['captions']) for r in vr}
        text=encode_texts(model,[t for r in vr for t in r['captions']],64,128,'cuda')
        gold=torch.tensor([i for i,r in enumerate(vr) for _ in r['captions']])
        check=None
        if args.smoke:
            indices=schedule['batches'][0][0]
            captions=torch.stack([tokens[records[i]['captions'][schedule['orders'][records[i]['video_id']][0]]] for i in indices])
            check=smoke_gradient_check(model,train_cache['features'][indices].cuda(),captions.cuda(),seed)
        write_json(folder/'preflight.json',dict(seed=seed,source_sha256=sources,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            initial_hashes=initial,initial_file_sha256=sha256_file(initial_path),
            schedule_sha256=sha256_file(folder/'schedule.json'),cache_error=error,smoke_checks=check,
            test_split_read=False,paper_ready=False,smoke_test=args.smoke,
            torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name()))
        results={}
        for arm in ('A','B'):
            results[arm]=run_arm(model,arm,seed,folder/arm,initial_path,initial,schedule,records,
                train_cache['features'],vr,vf,refs,text,gold,tokens,sources,pre,args.smoke)
        assert results['A']['trace_sha256']==results['B']['trace_sha256']
        a,b=(results[k]['evaluations'][-1] for k in ('A','B'))
        pair=dict(seed=seed,initial_hashes=initial,trace_match=True,smoke_test=args.smoke,
            decision=primary_decision(a['caption']['metrics'],b['caption']['metrics'],a['retrieval'],b['retrieval']),
            A_caption_gate=a['caption']['metrics']['primary_viability_gate']['pass'],
            results={k:str((folder/k/'result.json').resolve()) for k in ('A','B')})
        write_json(folder/'pair_result.json',pair)
        pairs.append(pair)
        print(json.dumps(pair),flush=True)
        del model,tokens,text
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output_root/'completion.json',dict(completed=True,seeds=list(SEEDS),pairs=pairs,
        smoke_test=args.smoke,paper_ready=False,test_split_read=False))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
