"""From-scratch pretraining of the Qwen-style MDLM and AR comparison arms."""
import os, json, math, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from textdiffusion.models.mhc_lite import MHCLiteBranchWrapper, StreamExpander, RMSNorm
from frr import VARIANTS, convert_model

import os as _os
DATA = _os.environ.get('PRETRAIN_DATA', 'data/pretrain_fwe_3600')
VOCAB = int(_os.environ.get('MODEL_VOCAB', 152064))
MASK_ID = int(_os.environ.get('MODEL_MASK_ID', 151666))
SEQ = int(_os.environ.get('MODEL_SEQ', 1024))
DIM = int(_os.environ.get('MODEL_DIM', 1024))
LAYERS = int(_os.environ.get('MODEL_LAYERS', 16))
HEADS = int(_os.environ.get('MODEL_HEADS', 16))
KV = int(_os.environ.get('MODEL_KV_HEADS', 4))
INTER = int(_os.environ.get('MODEL_INTERMEDIATE', 2816))
if DIM % HEADS or HEADS % KV:
    raise ValueError(f'invalid attention shape: dim={DIM}, heads={HEADS}, kv_heads={KV}')
HEAD_DIM = DIM // HEADS
# число потоков mHC и режим residual-микшера (learned = softmax по n! перестановкам;
# для n>=8 это неподъёмно, там нужен kromhc = кронекер из log2(n) 2x2 факторов)
STREAMS = int(_os.environ.get('MHC_STREAMS', 4))
RESMIX = _os.environ.get('MHC_RESMIX', 'learned')

def rope(x, base=10000.0):
    B,H,T,D = x.shape
    half = D//2
    freqs = 1.0/(base**(torch.arange(half, device=x.device, dtype=torch.float32)/half))
    t = torch.arange(T, device=x.device, dtype=torch.float32)
    ang = torch.outer(t, freqs)
    cos, sin = ang.cos()[None,None], ang.sin()[None,None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], -1)

class Attn(nn.Module):
    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal
        self.norm = RMSNorm(DIM)
        self.q = nn.Linear(DIM, DIM, bias=False)
        self.k = nn.Linear(DIM, KV*HEAD_DIM, bias=False)
        self.v = nn.Linear(DIM, KV*HEAD_DIM, bias=False)
        self.o = nn.Linear(DIM, DIM, bias=False)
    def forward(self, x):
        B,T,_ = x.shape
        x = self.norm(x)
        q = self.q(x).view(B,T,HEADS,HEAD_DIM).transpose(1,2)
        k = self.k(x).view(B,T,KV,HEAD_DIM).transpose(1,2)
        v = self.v(x).view(B,T,KV,HEAD_DIM).transpose(1,2)
        q, k = rope(q), rope(k)
        k = k.repeat_interleave(HEADS//KV, 1); v = v.repeat_interleave(HEADS//KV, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.o(y.transpose(1,2).reshape(B,T,DIM))

class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(DIM)
        self.gate = nn.Linear(DIM, INTER, bias=False)
        self.up = nn.Linear(DIM, INTER, bias=False)
        self.down = nn.Linear(INTER, DIM, bias=False)
    def forward(self, x):
        x = self.norm(x)
        return self.down(F.silu(self.gate(x)) * self.up(x))

class ScalarGate(nn.Module):
    def __init__(self, b=32):
        super().__init__()
        self.norm = RMSNorm(DIM)
        self.p = nn.Linear(DIM, b)
        self.w = nn.Linear(b, 1)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        nn.init.zeros_(self.w.weight); nn.init.zeros_(self.w.bias); nn.init.zeros_(self.p.bias)
        nn.init.xavier_uniform_(self.p.weight)
    def forward(self, h):
        z = F.silu(self.p(self.norm(h)))
        return 2.0*torch.sigmoid(self.alpha*self.w(z))  # [B,T,1], init == 1

class Block(nn.Module):
    def __init__(self, arm, causal=False):
        super().__init__()
        self.arm = arm
        self.attn, self.mlp = Attn(causal), Mlp()
        if arm == 'mhc':
            # MHC_PAPER=1: mHC-lite paper parametrization — sigmoid read, direct nC->n heads (no bottleneck)
            paper = _os.environ.get('MHC_PAPER') == '1'
            bd, rf = (None, 'sigmoid') if paper else (128, 'softmax')
            self.wa = MHCLiteBranchWrapper(DIM, self.attn, num_streams=STREAMS, bottleneck_dim=bd,
                                           read_function=rf, residual_mixing=RESMIX)
            self.wm = MHCLiteBranchWrapper(DIM, self.mlp, num_streams=STREAMS, bottleneck_dim=bd,
                                           read_function=rf, residual_mixing=RESMIX)
        elif arm == 'gate':
            self.ga, self.gm = ScalarGate(), ScalarGate()
    def forward(self, h, role_ids=None):
        if self.arm == 'frr':
            return self.wm(self.wa(h, role_ids), role_ids)
        if self.arm == 'mhc':
            return self.wm(self.wa(h))
        if self.arm == 'gate':
            h = h + self.ga(h)*self.attn(h)
            return h + self.gm(h)*self.mlp(h)
        h = h + self.attn(h)
        return h + self.mlp(h)

class Model(nn.Module):
    def __init__(self, arm, causal=False):
        super().__init__()
        variant = arm if arm in VARIANTS else None
        if variant:
            if STREAMS != 2 or causal or _os.environ.get('MHC_PAPER') != '1' or _os.environ.get('MHC_PROPER_INIT') != '1':
                raise ValueError('FRR requires MDLM, MHC_STREAMS=2, MHC_PAPER=1, MHC_PROPER_INIT=1')
            arm = 'mhc'
        self.arm = arm
        self.emb = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList([Block(arm, causal) for _ in range(LAYERS)])
        self.norm = RMSNorm(DIM)
        if arm == 'mhc':
            self.expander = StreamExpander(DIM, STREAMS, init_std=1e-6)
        self.apply(self._init)
        import os
        if os.environ.get('MHC_PROPER_INIT') == '1':
            from textdiffusion.models.mhc_lite import MHCLiteRouter as _R
            n = 0
            for mod in self.modules():
                if isinstance(mod, _R):
                    mod.reset_parameters(); n += 1
            print(f'proper mhc-lite init restored on {n} routers', flush=True)
        if os.environ.get('MHC_PAPER') == '1' and arm == 'mhc':
            # mHC-lite paper init: b_pre/b_post = -1 everywhere except one entry = +1;
            # the +1 position cycles by interface index (HC eq.14: e_{k mod n}).
            # Stream expansion = pure replication: no stream embeddings (zeroed + frozen).
            k = 0
            for blk in self.blocks:
                for w in (blk.wa, blk.wm):
                    nn.init.constant_(w.router.pre.bias, -1.0)
                    nn.init.constant_(w.router.post.bias, -1.0)
                    w.router.pre.bias.data[k % STREAMS] = 1.0
                    w.router.post.bias.data[k % STREAMS] = 1.0
                    k += 1
            self.expander.stream_embeddings.data.zero_()
            self.expander.stream_embeddings.requires_grad_(False)
            print(f'MHC_PAPER init applied on {k} interfaces (cyclic +1 bias, no stream embeddings)', flush=True)
        self.head = lambda h: F.linear(h, self.emb.weight)
        if variant:
            convert_model(self, variant)
    def _init(self, m):
        if isinstance(m, nn.Linear) and not hasattr(m, '_skip'):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    def forward(self, ids):
        role_ids = ids.eq(MASK_ID).long() if hasattr(self, 'frr_variant') else None
        h = self.emb(ids)
        if self.arm == 'mhc':
            h = self.expander(h)
        for b in self.blocks: h = b(h, role_ids)
        if self.arm == 'mhc':
            h = h.mean(-2)
        return self.head(self.norm(h))

def mdlm_batch(blocks, step, device):
    g = torch.Generator(device='cpu'); g.manual_seed(1337_000_000 + step)
    ids = blocks.to(device)
    B = ids.shape[0]
    t = torch.rand(B, generator=g).clamp_(0.02, 1.0).to(device)
    u = torch.rand(B, SEQ, generator=g).to(device)
    mask = u < t[:, None]
    mask[mask.sum(-1) == 0, 0] = True
    inp = ids.clone(); inp[mask] = MASK_ID
    return inp, ids, mask, t

def ar_loss(logits, ids):
    B, T, V = logits.shape
    fl = logits[:, :-1].reshape(-1, V); fi = ids[:, 1:].reshape(-1)
    ces = [F.cross_entropy(fl[i:i+8192], fi[i:i+8192], reduction='none').float()
           for i in range(0, fi.numel(), 8192)]
    return torch.cat(ces).mean()

def mdlm_loss(logits, ids, mask, t):
    B, T, V = logits.shape
    fl = logits.view(-1, V); fi = ids.view(-1)
    idx = mask.view(-1).nonzero(as_tuple=True)[0]
    ces = [F.cross_entropy(fl[idx[i:i+8192]], fi[idx[i:i+8192]], reduction='none').float()
           for i in range(0, idx.numel(), 8192)]
    ce = torch.cat(ces)
    per_seq = torch.zeros(B, device=logits.device, dtype=torch.float32).index_add_(0, (idx//T), ce)
    return (per_seq / T / t).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['baseline','mhc','gate', *VARIANTS], required=True)
    ap.add_argument('--steps', type=int, default=16000)
    ap.add_argument('--micro', type=int, default=32)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--router-lr', type=float, default=1e-3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--objective', choices=['mdlm','ar'], default='mdlm')
    ap.add_argument('--val-every', type=int, default=500)
    ap.add_argument('--val-micro', type=int, default=16)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--warmup-steps', type=int, default=300)
    ap.add_argument('--target-tokens', type=int, help='Override --steps with ceil(target/tokens_per_update)')
    ap.add_argument('--stop-after', type=int, help='Pause after this many updates without changing the LR schedule')
    ap.add_argument('--keep-every', type=int, default=5000, help='Retain milestone checkpoints; 0 keeps only the latest')
    a = ap.parse_args()
    RANK = int(os.environ.get('RANK', 0)); WORLD = int(os.environ.get('WORLD_SIZE', 1))
    LR_ = int(os.environ.get('LOCAL_RANK', 0))
    if WORLD > 1:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(LR_)
    is_main = (RANK == 0)
    tokens_per_update = a.micro * a.accum * SEQ * WORLD
    if a.target_tokens is not None:
        if a.target_tokens < 1:
            raise ValueError('--target-tokens must be positive')
        a.steps = math.ceil(a.target_tokens / tokens_per_update)
    if is_main: os.makedirs(a.out, exist_ok=True)
    if a.arm in VARIANTS:
        from frr_experiment import record_config
        record_config(a, WORLD)
    torch.manual_seed(1337)
    dev = f'cuda:{LR_}' if WORLD > 1 else 'cuda'
    meta = json.load(open(f'{DATA}/meta.json'))
    nval = meta['n_val_blocks']; nb = meta['n_blocks']
    arr = np.load(f'{DATA}/all_tokens.npy', mmap_mode='r')
    model = Model(a.arm, causal=(a.objective=='ar')).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    if is_main:
        run_config = {
            'arm': a.arm, 'objective': a.objective, 'seed': 1337,
            'architecture': {'vocab': VOCAB, 'seq': SEQ, 'dim': DIM, 'layers': LAYERS,
                             'heads': HEADS, 'kv_heads': KV, 'head_dim': HEAD_DIM,
                             'intermediate': INTER, 'streams': STREAMS if a.arm in ('mhc', *VARIANTS) else 1,
                             'residual_mixing': RESMIX if a.arm == 'mhc' else None},
            'parameters': nparams, 'world_size': WORLD, 'micro_batch_per_rank': a.micro,
            'gradient_accumulation': a.accum, 'tokens_per_update': tokens_per_update,
            'target_tokens': a.target_tokens, 'updates': a.steps,
            'planned_input_tokens': a.steps * tokens_per_update,
            'optimizer': {'name': 'AdamW', 'lr': a.lr, 'betas': [0.9, 0.95],
                          'weight_decay_2d': 0.1, 'warmup_steps': a.warmup_steps,
                          'schedule': 'cosine_to_0.1x'},
            'data': DATA, 'val_every': a.val_every, 'val_micro': a.val_micro,
            'save_every': a.save_every, 'keep_every': a.keep_every,
        }
        config_path = f'{a.out}/run_config.json'
        if os.path.exists(config_path):
            prior = json.load(open(config_path))
            if prior != run_config:
                raise ValueError(f'run config mismatch at {config_path}')
        else:
            with open(config_path+'.tmp', 'w') as target:
                json.dump(run_config, target, indent=2)
            os.replace(config_path+'.tmp', config_path)
        print(f'arm={a.arm} params={nparams/1e6:.1f}M world={WORLD} '
              f'tokens/update={tokens_per_update} updates={a.steps} planned_tokens={a.steps*tokens_per_update}'
              + (f' streams={STREAMS} resmix={RESMIX}' if a.arm == 'mhc' else ''), flush=True)
    raw_model = model
    if WORLD > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LR_])
    router_keys = ('router', 'expander', 'ga.', 'gm.', '.p.', '.w.', 'alpha')
    if _os.environ.get('MHC_PAPER') == '1':
        # papers use one lr for everything; standard nanoGPT-style wd: 2D params decay, 1D don't
        dec = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nod = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        opt = torch.optim.AdamW([
            {'params': dec, 'lr': a.lr, 'weight_decay': 0.1},
            {'params': nod, 'lr': a.lr, 'weight_decay': 0.0}],
            betas=(0.9, 0.95), fused=True)
        a.router_lr = a.lr  # keep the lr-schedule line consistent
    else:
        rp = [p for n,p in model.named_parameters() if any(k in n for k in ('wa.router','wm.router','expander','.ga.','.gm.'))]
        rset = {id(p) for p in rp}
        bp = [p for p in model.parameters() if id(p) not in rset]
        opt = torch.optim.AdamW([
            {'params': bp, 'lr': a.lr, 'weight_decay': 0.1},
            {'params': rp, 'lr': a.router_lr, 'weight_decay': 0.0}],
            betas=(0.9, 0.95), fused=True)
    warm = a.warmup_steps
    def lr_mult(s):
        if s < warm: return s/warm
        p = (s-warm)/max(a.steps-warm, 1)
        return 0.1 + 0.45*(1+math.cos(math.pi*p))*0.9 if False else 0.1 + 0.9*0.5*(1+math.cos(math.pi*p))
    perm = np.random.default_rng(1337).permutation(nb - nval) + nval
    mlog = open(f'{a.out}/metrics.jsonl', 'a')
    # fixed val
    val_blocks = torch.from_numpy(arr[:64].astype(np.int64))
    def validate():
        model.eval(); res = {}
        if a.objective == 'ar':
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                losses = []
                for i in range(0, 64, a.val_micro):
                    ids = val_blocks[i:i+a.val_micro].to(dev)
                    losses.append(float(ar_loss(model(ids), ids)))
            res['val_mean'] = sum(losses)/len(losses)
            model.train(); return res
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            for ri, r in enumerate([0.1,0.3,0.5,0.7,0.9]):
                g = torch.Generator(); g.manual_seed(777+ri)
                losses = []
                for i in range(0, 64, a.val_micro):
                    ids = val_blocks[i:i+a.val_micro].to(dev)
                    u = torch.rand(ids.shape, generator=g).to(dev)
                    mask = u < r; mask[mask.sum(-1)==0, 0] = True
                    inp = ids.clone(); inp[mask] = MASK_ID
                    tt = torch.full((ids.shape[0],), r, device=dev)
                    losses.append(float(mdlm_loss(model(inp), ids, mask, tt)))
                res[f'r{r}'] = sum(losses)/len(losses)
        res['val_mean'] = sum(res.values())/5
        model.train(); return res
    step0 = 0
    import glob
    cks = sorted(glob.glob(f'{a.out}/ckpt_*.pt'), key=lambda p: int(p.split('_')[-1][:-3]))
    # обрыв на записи чекпоинта оставляет битый zip; берём самый свежий читаемый,
    # иначе обёртка-рестартер уходит в бесконечный краш-луп
    while cks:
        try:
            st = torch.load(cks[-1], map_location='cpu', weights_only=False)
        except Exception as e:
            if is_main: print(f'SKIP corrupt checkpoint {cks[-1]}: {e}', flush=True)
            cks.pop(); continue
        raw_model.load_state_dict(st['model'])
        if 'opt' in st: opt.load_state_dict(st['opt'])
        step0 = st['step'] + 1
        if is_main: print(f'RESUMED from {cks[-1]} at step {step0}', flush=True)
        break
    if a.arm in VARIANTS:
        from frr_experiment import diagnostics
        if step0 == 0:
            diagnostics(raw_model, val_blocks, dev, a.out, -1)
    t0 = time.time(); seen = 0
    model.train()
    end_step = min(a.steps, a.stop_after) if a.stop_after else a.steps
    for step in range(step0, end_step):
        if a.arm in VARIANTS:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            step_started = time.perf_counter()
        for g in opt.param_groups: g['lr'] = (a.lr if g['weight_decay']>0 else a.router_lr)*lr_mult(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for acc in range(a.accum):
            pos = ((step*a.accum+acc)*WORLD + RANK)*a.micro % (nb-nval-a.micro)
            idx = perm[pos:][:a.micro]
            blocks = torch.from_numpy(arr[np.sort(idx)].astype(np.int64))
            if a.objective == 'ar':
                ids = blocks.to(dev)
                with torch.autocast('cuda', torch.bfloat16):
                    loss = ar_loss(model(ids), ids) / a.accum
            else:
                inp, ids, mask, t = mdlm_batch(blocks, (step*a.accum+acc)*WORLD + RANK, dev)
                with torch.autocast('cuda', torch.bfloat16):
                    loss = mdlm_loss(model(inp), ids, mask, t) / a.accum
            loss.backward(); tot += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(tot):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        opt.step()
        cost = {}
        if a.arm in VARIANTS:
            torch.cuda.synchronize()
            cost = {'step_seconds': time.perf_counter()-step_started,
                    'train_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'gpu_name': torch.cuda.get_device_name()}
        seen += a.micro*a.accum*SEQ*WORLD
        if (step % 20 == 0 or step == end_step-1) and is_main:
            el = time.time()-t0
            print(f'step {step} loss {tot:.4f} tok/s {seen/max(el,1):.0f}', flush=True)
            mlog.write(json.dumps({'step':step,'loss':tot,'toks':seen,
                                   'total_input_tokens': (step+1)*tokens_per_update, **cost})+'\n'); mlog.flush()
        if (step % a.val_every == 0 or step == end_step-1) and is_main:
            v = validate(); v['step']=step
            print('VAL', json.dumps(v), flush=True)
            mlog.write(json.dumps(v)+'\n'); mlog.flush()
            if a.arm in VARIANTS:
                diagnostics(raw_model, val_blocks, dev, a.out, step)
        if ((step % a.save_every == 0 and step>0) or step == end_step-1) and is_main:
            target = f'{a.out}/ckpt_{step}.pt'
            torch.save({'model': raw_model.state_dict(), 'opt': opt.state_dict(), 'step': step, 'arm': a.arm}, target+'.tmp')
            os.replace(target+'.tmp', target)
            import glob as _gl
            for old in sorted(_gl.glob(f'{a.out}/ckpt_*.pt'), key=lambda p:int(p.split('_')[-1][:-3]))[:-1]:
                st_ = int(old.split('_')[-1][:-3])
                if (not a.keep_every or st_ % a.keep_every != 0) and st_ != a.steps-1:
                    os.remove(old)
    print('DONE' if end_step == a.steps else f'PAUSED at {end_step}', flush=True)

if __name__ == '__main__':
    main()
