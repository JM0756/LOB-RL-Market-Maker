"""
train_agent.py
──────────────
Training and evaluation script for the RL market-making agent.

Trains a PPO agent on MarketMakerEnv using Stable Baselines3, saves the
model, then runs a head-to-head evaluation comparing the trained agent
against a random-action baseline so you can immediately verify learning.

USAGE
─────
    # Train, save, and evaluate in one shot:
    python train_agent.py

    # Skip training and jump straight to evaluation (model must exist):
    python train_agent.py --eval-only

    # Override total timesteps from the command line:
    python train_agent.py --timesteps 500000

DEPENDENCIES
────────────
    pip install stable-baselines3 gymnasium numpy

    lob_engine must be compiled and on PYTHONPATH — see CMakeLists.txt.

OUTPUT FILES
────────────
    ppo_market_maker.zip          ← trained model weights (SB3 format)
    logs/PPO_market_maker_*/      ← TensorBoard logs (optional, see below)

TENSORBOARD
───────────
    tensorboard --logdir logs/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure local modules (MarketMakerEnv, lob_engine) are importable
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np

# ── Stable Baselines3 ─────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.evaluation import evaluate_policy

# ── Local environment ─────────────────────────────────────────────────────────
from MarketMakerEnv import MarketMakerEnv


# =============================================================================
# CONFIGURATION
# =============================================================================

# ── File paths ────────────────────────────────────────────────────────────────
MODEL_PATH      = "ppo_market_maker"          # .zip appended automatically by SB3
VECNORM_PATH    = "ppo_market_maker_vecnorm.pkl"  # saved VecNormalize statistics
CHECKPOINT_DIR  = "checkpoints/"             # intermediate model saves
LOG_DIR         = "logs/"                    # TensorBoard log directory

# ── Training ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS   = 250_000   # total env steps across all parallel envs
N_ENVS            = 4         # number of parallel environments
                               # PPO collects N_ENVS × N_STEPS transitions per
                               # update, so effective batch = 4 × 2048 = 8192

# ── PPO Hyperparameters ───────────────────────────────────────────────────────
#
#  These defaults work well for low-dimensional Box observations with discrete
#  actions. Tune if you extend the observation space or change the reward scale.
#
#  n_steps      : rollout buffer length per env. Larger = more on-policy data
#                 per update, more stable but slower. 2048 is the SB3 default.
#  batch_size   : minibatch size for each gradient update. Must divide
#                 (n_steps × n_envs) = 8192 evenly. 256 gives 32 minibatches.
#  n_epochs     : how many passes over the rollout buffer per PPO update.
#                 More epochs = higher data efficiency but risk of over-fitting.
#  gamma        : discount factor. 0.99 is standard; lower (e.g. 0.95) makes
#                 the agent more myopic — can help if rewards are very dense.
#  gae_lambda   : GAE smoothing parameter. 0.95 balances bias vs variance in
#                 advantage estimation.
#  clip_range   : PPO surrogate objective clip. 0.2 is the canonical default.
#  ent_coef     : entropy bonus coefficient. 0.005 gently encourages
#                 exploration of all four actions early in training, preventing
#                 premature convergence to a single action (e.g. always Hold).
#  vf_coef      : value function loss weight relative to policy loss.
#  max_grad_norm: gradient clipping threshold — prevents exploding gradients.
#  learning_rate: Adam step size. 3e-4 is the SB3 default and works here
#                 because VecNormalize keeps reward variance near 1.

PPO_HYPERPARAMS: dict[str, Any] = {
    "n_steps"       : 2048,
    "batch_size"    : 256,
    "n_epochs"      : 10,
    "gamma"         : 0.99,
    "gae_lambda"    : 0.95,
    "clip_range"    : 0.2,
    "ent_coef"      : 0.005,
    "vf_coef"       : 0.5,
    "max_grad_norm" : 0.5,
    "learning_rate" : 3e-4,
}

# ── Policy network architecture ───────────────────────────────────────────────
#
#  Two hidden layers of 64 units each, shared between policy and value heads.
#  The observation is 9-dimensional and the action space is Discrete(4), so
#  64 units is comfortably expressive without being over-parameterised.
#  For a deeper look at when to increase capacity, see SB3 tips:
#  https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

POLICY_KWARGS: dict[str, Any] = {
    "net_arch": [64, 64],
}

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_EPISODES    = 10     # number of episodes for post-training evaluation
EVAL_FREQ        = 10_000 # steps between in-training EvalCallback checks
CHECKPOINT_FREQ  = 50_000 # steps between CheckpointCallback saves
RANDOM_SEED      = 42


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_env(rank: int = 0, seed: int = RANDOM_SEED) -> MarketMakerEnv:
    """
    Factory function that creates one wrapped MarketMakerEnv instance.

    Wrapping with Monitor is mandatory for SB3: it records episode rewards
    and lengths so EvalCallback and the SB3 logger can report meaningful
    metrics. Without Monitor, SB3 would raise a warning and episode statistics
    would be missing from TensorBoard.

    The `rank` offset ensures each parallel env has a different random seed,
    which prevents all N_ENVS environments from producing identical rollouts
    (correlated trajectories would reduce the effective batch diversity).
    """
    def _init() -> MarketMakerEnv:
        env = MarketMakerEnv(instrument_id=f"SIM_{rank}")
        env = Monitor(env)           # records ep_rew_mean, ep_len_mean
        env.reset(seed=seed + rank)
        return env
    return _init


# =============================================================================
# TRAINING
# =============================================================================

def train(total_timesteps: int = TOTAL_TIMESTEPS) -> PPO:
    """
    Build the vectorised environment, instantiate PPO, attach callbacks,
    and run the training loop.

    Environment stack
    ─────────────────
    Raw env  →  Monitor  →  VecEnv (4 parallel)  →  VecNormalize
                                                        ↑
                                        Normalises observations and rewards
                                        to zero-mean / unit-variance using
                                        running statistics updated each step.
                                        This is the single most impactful
                                        change for stabilising PPO training
                                        on this environment because:
                                          - mid_price obs  ≈ 10 000 (cents)
                                          - volume obs     ≈ 20–100
                                          - inventory obs  ≈ ±200
                                        Without normalisation the policy
                                        gradient is dominated by the large
                                        mid-price dimension.

    Callbacks
    ─────────
    EvalCallback     : runs EVAL_EPISODES deterministic episodes every
                       EVAL_FREQ steps and saves the best model seen so far
                       as 'best_model.zip'.
    CheckpointCallback : saves a full checkpoint every CHECKPOINT_FREQ steps
                         so training can be resumed if interrupted.
    """

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)

    print("=" * 65)
    print("  Market Maker RL — PPO Training")
    print("=" * 65)
    print(f"  Total timesteps  : {total_timesteps:,}")
    print(f"  Parallel envs    : {N_ENVS}")
    print(f"  Effective batch  : {N_ENVS * PPO_HYPERPARAMS['n_steps']:,} transitions")
    print(f"  Model output     : {MODEL_PATH}.zip")
    print("=" * 65, "\n")

    # ── Vectorised training environment ───────────────────────────────────────
    train_vec_env = DummyVecEnv([make_env(rank=i) for i in range(N_ENVS)])

    # VecNormalize wraps the entire VecEnv.
    # norm_obs=True  : normalise observations using running mean/std.
    # norm_reward=True: normalise rewards using running variance (not mean).
    #                   This keeps the value function target near unit scale
    #                   regardless of the absolute reward magnitude.
    # clip_obs=10.0  : clip normalised observations to [-10, 10] to prevent
    #                   very rare extreme values from corrupting the running
    #                   statistics during early training.
    train_env = VecNormalize(
        train_vec_env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0,
        gamma       = PPO_HYPERPARAMS["gamma"],
    )

    # ── Separate evaluation environment ───────────────────────────────────────
    #
    #  EvalCallback needs its own environment so it can run deterministic
    #  rollouts without interfering with the training rollout buffer.
    #  Crucially, we do NOT wrap this env with VecNormalize — instead we
    #  sync its statistics from the training VecNormalize before each eval
    #  via EvalCallback's `eval_env` parameter and `sync_envs_normalization`.
    eval_vec_env = DummyVecEnv([make_env(rank=100, seed=RANDOM_SEED + 1000)])
    eval_env = VecNormalize(
        eval_vec_env,
        norm_obs    = True,
        norm_reward = False,    # do NOT normalise rewards during eval —
                                # we want raw reward values for fair comparison
        clip_obs    = 10.0,
        gamma       = PPO_HYPERPARAMS["gamma"],
        training    = False,    # freeze running stats during evaluation
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = "./",          # saves best_model.zip here
        log_path             = LOG_DIR,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,          # no action sampling during eval
        render               = False,
        verbose              = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "ppo_market_maker_ckpt",
        verbose     = 1,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ── PPO model ─────────────────────────────────────────────────────────────
    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        policy_kwargs = POLICY_KWARGS,
        tensorboard_log = LOG_DIR,
        verbose       = 1,
        seed          = RANDOM_SEED,
        **PPO_HYPERPARAMS,
    )

    print(f"Policy architecture:\n{model.policy}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    t0 = time.time()

    model.learn(
        total_timesteps   = total_timesteps,
        callback          = callbacks,
        tb_log_name       = "PPO_market_maker",
        reset_num_timesteps = True,
        progress_bar      = True,   # requires tqdm: pip install tqdm
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s "
          f"({total_timesteps / elapsed:,.0f} steps/sec)\n")

    # ── Persist model and normalisation statistics ─────────────────────────────
    #
    #  IMPORTANT: Save VecNormalize statistics alongside the model weights.
    #  If you load the model later without the correct running mean/std, the
    #  normalised observations will be on a completely different scale from
    #  training and the policy will produce garbage outputs.
    model.save(MODEL_PATH)
    train_env.save(VECNORM_PATH)

    print(f"Model saved to:             {MODEL_PATH}.zip")
    print(f"VecNormalize stats saved to: {VECNORM_PATH}\n")

    train_env.close()
    eval_env.close()

    return model


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def _run_episode(
    model: PPO,
    env:   MarketMakerEnv,
    deterministic: bool = True,
    render: bool = False,
) -> dict[str, float]:
    """
    Run a single episode with the given model and return a stats dictionary.

    Parameters
    ──────────
    model         : trained SB3 PPO model (or None to use random actions).
    env           : unwrapped MarketMakerEnv instance (no VecNormalize).
    deterministic : if True, use argmax policy; if False, sample from the
                    policy distribution (adds stochasticity at test time).
    render        : if True, call env.render() each step.

    Returns
    ───────
    dict with keys: total_reward, total_steps, final_wealth, final_inventory,
                    final_cash, action_counts, terminated.
    """
    obs, _ = env.reset()
    total_reward   = 0.0
    total_steps    = 0
    action_counts  = {0: 0, 1: 0, 2: 0, 3: 0}
    info           = {}

    while True:
        if model is not None:
            # model.predict expects a numpy array; unwrapped env gives one.
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward  += reward
        total_steps   += 1
        action_counts[int(action)] += 1

        if render:
            env.render()

        if terminated or truncated:
            break

    return {
        "total_reward"    : total_reward,
        "total_steps"     : total_steps,
        "final_wealth"    : info.get("total_wealth", 0.0),
        "final_inventory" : info.get("inventory",    0),
        "final_cash"      : info.get("cash",         0.0),
        "action_counts"   : action_counts,
        "terminated"      : terminated,
    }


def _print_episode_stats(
    label: str,
    results: list[dict],
    initial_mid: int = 10_000,
) -> None:
    """Pretty-print summary statistics over multiple evaluation episodes."""

    rewards    = [r["total_reward"]    for r in results]
    wealth     = [r["final_wealth"]    for r in results]
    inventory  = [r["final_inventory"] for r in results]
    steps      = [r["total_steps"]     for r in results]

    # Aggregate action distribution across all episodes.
    agg_actions: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        for a, cnt in r["action_counts"].items():
            agg_actions[a] += cnt
    total_actions = sum(agg_actions.values())
    action_names  = {0: "Hold/Cancel", 1: "Quote Tight",
                     2: "Quote Wide",  3: "Clear Inv."}

    print(f"\n{'─' * 65}")
    print(f"  {label}  ({len(results)} episodes)")
    print(f"{'─' * 65}")
    print(f"  Cumulative Reward  : "
          f"mean={np.mean(rewards):+.2f}  "
          f"std={np.std(rewards):.2f}  "
          f"min={np.min(rewards):+.2f}  "
          f"max={np.max(rewards):+.2f}")
    print(f"  Final Wealth (¢)   : "
          f"mean={np.mean(wealth):+.1f}  "
          f"std={np.std(wealth):.1f}")
    print(f"  Final Wealth ($)   : "
          f"mean=${np.mean(wealth)/100:+.2f}  "
          f"std=${np.std(wealth)/100:.2f}")
    print(f"  Final Inventory    : "
          f"mean={np.mean(inventory):+.1f}  "
          f"std={np.std(inventory):.1f}")
    print(f"  Episode Length     : "
          f"mean={np.mean(steps):.1f}  "
          f"(terminated={sum(r['terminated'] for r in results)}/{len(results)})")
    print(f"  Action Distribution:")
    for a in range(4):
        pct = 100 * agg_actions[a] / max(total_actions, 1)
        bar = "█" * int(pct / 2)
        print(f"    [{a}] {action_names[a]:<14s} {pct:5.1f}%  {bar}")
    print(f"{'─' * 65}")


# =============================================================================
# EVALUATION MAIN
# =============================================================================

def evaluate(
    n_episodes: int = EVAL_EPISODES,
    render_one: bool = False,
) -> None:
    """
    Load the saved model and VecNormalize statistics, then run two evaluation
    suites side-by-side:

      1. Trained PPO agent  — deterministic policy (argmax actions)
      2. Random baseline    — uniform random action each step

    The comparison is run on raw (un-normalised) observations via an unwrapped
    MarketMakerEnv so that reward and wealth values are in original cent units.

    Note on VecNormalize at inference time
    ──────────────────────────────────────
    The PPO policy was trained on *normalised* observations. At inference we
    must apply the same normalisation transform before passing observations to
    the policy. We do this by:
      1. Wrapping the eval env in a single-env VecNormalize.
      2. Loading the saved running statistics with .load().
      3. Setting training=False so the statistics are frozen (not updated).

    For the random baseline we skip normalisation entirely because the random
    policy doesn't use the observation at all.
    """

    print("\n" + "=" * 65)
    print("  Post-Training Evaluation")
    print("=" * 65)

    # ── Check saved artefacts exist ───────────────────────────────────────────
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}.zip' not found. "
            "Run training first (remove --eval-only flag)."
        )

    # ── Build a normalised single-env wrapper for the PPO agent ───────────────
    #
    #  make_vec_env creates a VecEnv with 1 environment. VecNormalize.load()
    #  restores the running mean/std computed during training. Setting
    #  training=False ensures inference doesn't update those statistics.
    eval_vec_env = DummyVecEnv([make_env(rank=200, seed=RANDOM_SEED + 9999)])

    eval_env_normalised = VecNormalize.load(VECNORM_PATH, eval_vec_env)
    eval_env_normalised.training    = False   # freeze running stats
    eval_env_normalised.norm_reward = False   # return raw rewards

    # Load model; attach the normalised eval env so model.predict() sees
    # correctly scaled observations.
    model = PPO.load(MODEL_PATH, env=eval_env_normalised)
    print(f"\nLoaded model from:  {MODEL_PATH}.zip")
    print(f"Loaded VecNorm from: {VECNORM_PATH}")
    print(f"Running {n_episodes} evaluation episodes...\n")

    # ── Evaluate the trained agent ─────────────────────────────────────────────
    #
    #  We use SB3's evaluate_policy() for the headline mean/std numbers (it
    #  handles the VecEnv interface correctly), then run our own loop for the
    #  detailed per-episode breakdown and action distribution.
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env_normalised,
        n_eval_episodes = n_episodes,
        deterministic   = True,
        render          = False,
    )
    print(f"[SB3 evaluate_policy]  "
          f"Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")

    # Now run the same episodes through the *unwrapped* environment so we can
    # collect full info dicts (wealth, inventory, cash) and action distributions.
    # The model's policy was trained on normalised obs; we replicate normalisation
    # here by manually calling eval_env_normalised.normalize_obs() on each step.
    ppo_results: list[dict] = []
    unwrapped_env = MarketMakerEnv(instrument_id="EVAL_PPO", render_mode="human" if render_one else None)

    for ep in range(n_episodes):
        render_this = render_one and (ep == 0)   # only render the first episode
        obs, _ = unwrapped_env.reset(seed=RANDOM_SEED + ep)
        total_reward  = 0.0
        total_steps   = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        info          = {}

        while True:
            # Normalise the observation the same way the training env did.
            obs_normalised = eval_env_normalised.normalize_obs(
                obs.reshape(1, -1).astype(np.float32)
            )
            
            # Predict the action
            action, _ = model.predict(obs_normalised, deterministic=True)
            
            # Extract the scalar value from the numpy array!
            scalar_action = int(action.item())

            obs, reward, terminated, truncated, info = unwrapped_env.step(scalar_action)
            total_reward  += reward
            total_steps   += 1
            action_counts[scalar_action] += 1

            if render_this:
                unwrapped_env.render()

            if terminated or truncated:
                break

        ppo_results.append({
            "total_reward"    : total_reward,
            "total_steps"     : total_steps,
            "final_wealth"    : info.get("total_wealth", 0.0),
            "final_inventory" : info.get("inventory",    0),
            "final_cash"      : info.get("cash",         0.0),
            "action_counts"   : action_counts,
            "terminated"      : terminated,
        })

    unwrapped_env.close()
    eval_env_normalised.close()

    # ── Random baseline ───────────────────────────────────────────────────────
    random_results: list[dict] = []
    baseline_env = MarketMakerEnv(instrument_id="EVAL_RAND")

    for ep in range(n_episodes):
        obs, _ = baseline_env.reset(seed=RANDOM_SEED + ep)   # same seeds as PPO
        total_reward  = 0.0
        total_steps   = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        info          = {}

        while True:
            action = baseline_env.action_space.sample()
            obs, reward, terminated, truncated, info = baseline_env.step(int(action))
            total_reward  += reward
            total_steps   += 1
            action_counts[int(action)] += 1

            if terminated or truncated:
                break

        random_results.append({
            "total_reward"    : total_reward,
            "total_steps"     : total_steps,
            "final_wealth"    : info.get("total_wealth", 0.0),
            "final_inventory" : info.get("inventory",    0),
            "final_cash"      : info.get("cash",         0.0),
            "action_counts"   : action_counts,
            "terminated"      : terminated,
        })

    baseline_env.close()

    # ── Print side-by-side results ─────────────────────────────────────────────
    _print_episode_stats("PPO Agent    (deterministic)", ppo_results)
    _print_episode_stats("Random Agent (uniform random)", random_results)

    # ── Head-to-head verdict ───────────────────────────────────────────────────
    ppo_mean    = np.mean([r["total_reward"] for r in ppo_results])
    random_mean = np.mean([r["total_reward"] for r in random_results])
    delta       = ppo_mean - random_mean
    win_rate    = np.mean([
        p["total_reward"] > r["total_reward"]
        for p, r in zip(ppo_results, random_results)
    ])

    print(f"\n{'═' * 65}")
    print(f"  HEAD-TO-HEAD VERDICT  ({n_episodes} matched episodes, same seeds)")
    print(f"{'═' * 65}")
    print(f"  PPO mean reward    : {ppo_mean:+.4f}")
    print(f"  Random mean reward : {random_mean:+.4f}")
    print(f"  Delta (PPO − Rand) : {delta:+.4f}")
    print(f"  PPO win rate       : {win_rate * 100:.1f}%")

    if delta > 0 and win_rate >= 0.6:
        print("\n  ✅  PPO beats the random baseline — learning confirmed.")
    elif delta > 0:
        print("\n  ⚠️   PPO has a higher mean but inconsistent wins. "
              "Consider more training.")
    else:
        print("\n  ❌  PPO does not beat the random baseline. "
              "Review hyperparameters or reward shaping.")
    print(f"{'═' * 65}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate the PPO market-making agent."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and go straight to evaluation "
             "(requires a saved model at ppo_market_maker.zip).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,}).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=EVAL_EPISODES,
        help=f"Number of evaluation episodes (default: {EVAL_EPISODES}).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the first evaluation episode step-by-step.",
    )
    args = parser.parse_args()

    if not args.eval_only:
        train(total_timesteps=args.timesteps)

    evaluate(n_episodes=args.eval_episodes, render_one=args.render)


if __name__ == "__main__":
    main()
