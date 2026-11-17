"""Same-input saved-policy comparison against the real original SB3/CNN."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from native_ppo import load_native_agent


def main():
    batch = Path(__file__).resolve().parents[1]
    old = Path('/home/user/navigation_project/a_pipeline/methods/experiments/drl_vo_ros2_offline')
    sys.path.insert(0, str(old))
    from drlvo_model import load_policy_strict
    inputs = Path('/home/user/workspace/social_navigation_runtime_20261010/evidence/drlvo_latestmap_3cm_20261016T0130MYT/runs/long_segment_1/policy_inputs')
    files = sorted(inputs.glob('*.npy'))
    selected = [files[i] for i in np.linspace(0, len(files)-1, 24, dtype=int)]
    observations = [np.load(f) for f in selected]
    torch.set_num_threads(2)
    models = Path('/home/user/navigation_project/a_pipeline/github_src/drl_vo_nav-drl_vo/drl_vo/src/model')
    results = []
    for name in ('drl_vo.zip', 'drl_pre_train.zip', 'drl.zip'):
        reference = load_native_agent(models/name)
        rebuilt, _ = load_policy_strict(models/name)
        errors, means, stds = [], [], []
        reference.policy.eval()
        rebuilt.eval()
        with torch.no_grad():
            for obs in observations:
                x = torch.from_numpy(obs[None])
                pi, vf, _ = reference.policy._get_latent(x)
                expected = reference.policy.action_net(pi)
                actual, value = rebuilt(x)
                errors.append(float((expected-actual).abs().max()))
                means.append(expected[0].tolist())
                stds.append(reference.policy.log_std.exp().tolist())
            blank = torch.zeros(1,19202)
            pi, _, _ = reference.policy._get_latent(blank)
            blank_mean = reference.policy.action_net(pi)[0].tolist()
        means = np.asarray(means)
        assert max(errors)<2e-5
        results.append(dict(model=name,sha256=hashlib.sha256((models/name).read_bytes()).hexdigest(),
            actual_original_SB3='0.10.0', original_custom_cnn_source=str(batch/'reference/custom_cnn_full.py'),
            cases=len(selected), max_raw_mean_abs_error=max(errors),
            linear_mean_min=float(means[:,0].min()),linear_mean_max=float(means[:,0].max()),
            mean_linear_outside_action_box_fraction=float((np.abs(means[:,0])>1).mean()),
            zero_observation_raw_mean=blank_mean, action_std=stds[0]))
    report = dict(status='ORIGINAL_NATIVE_SAME_INPUT_EQUIVALENCE_PASS_SATURATION_PERSISTS',
        observations=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in selected],
        models=results, reconstructed_loader_mismatch_observed=False,
        saturation_present_before_NewMetro_updates=True, why_original_checkpoint_learned_this='UNDETERMINED',
        observation_distribution_cause='NOT_RULED_OUT', action_head_rescaled=False,
        weights_reinitialized=False, training_convergence='NOT_CLAIMED')
    (batch/'audit/ACTION_DIAGNOSIS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='observations'},ensure_ascii=False))


if __name__ == '__main__': main()
