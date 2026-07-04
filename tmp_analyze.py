import json, glob, os, yaml

base = r'C:\Users\assaf\Desktop\Coding\Msc\Master\NCC-PINN\kdv_00_smoketest_20260606_210856\kdv-2-37-37-37-1-tanh'

results = []
for run_dir in sorted(os.listdir(base)):
    full = os.path.join(base, run_dir)
    mf = os.path.join(full, 'metrics.json')
    cf = os.path.join(full, 'config_used.yaml')
    if not os.path.exists(mf):
        continue

    with open(mf) as f:
        m = json.load(f)
    with open(cf) as f:
        cfg = yaml.safe_load(f)

    epochs = m['epochs']
    rel_l2 = m['eval_rel_l2']

    # second-to-last = last Phase 1 eval (last entry is Phase 3)
    pre_spawn_epoch = epochs[-2]
    pre_spawn_l2 = rel_l2[-2]
    phase1_l2 = rel_l2[:-1]
    best_l2 = min(phase1_l2)
    best_epoch = epochs[phase1_l2.index(best_l2)]

    name = cfg.get('name', run_dir)
    kdv = cfg.get('kdv', {})
    causal = kdv.get('causal_training', {})
    ff = kdv.get('fourier_features', {})
    lra = kdv.get('lra', {})

    results.append({
        'name': name, 'ts': run_dir,
        'pre_spawn_l2': pre_spawn_l2, 'pre_spawn_epoch': pre_spawn_epoch,
        'best_l2': best_l2, 'best_epoch': best_epoch,
        'causal': causal.get('enabled', False),
        'tol': causal.get('tol_schedule', []),
        'rwf': kdv.get('rwf', False),
        'ff': ff.get('enabled', False),
        'lra': lra.get('enabled', False),
    })

results.sort(key=lambda x: x['pre_spawn_l2'])

print('Ranked by rel-L2 at last Phase-1 epoch (before spawning):')
print()
for i, r in enumerate(results, 1):
    n = r['name']
    ts = r['ts']
    pl2 = r['pre_spawn_l2']
    pe = r['pre_spawn_epoch']
    bl2 = r['best_l2']
    be = r['best_epoch']
    causal = r['causal']
    tol = r['tol']
    rwf = r['rwf']
    ff = r['ff']
    lra = r['lra']
    print(f'{i}. {n} ({ts})')
    print(f'   pre-spawn rel-L2 = {pl2:.4f} (epoch {pe})  |  best = {bl2:.4f} (epoch {be})')
    print(f'   causal={causal} tol={tol}  rwf={rwf}  fourier={ff}  lra={lra}')
    print()
