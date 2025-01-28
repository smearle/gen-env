import collections
from functools import partial
import glob
import os
import shutil
import sys
from timeit import default_timer as timer
import traceback
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import chex
import flax
from flax import linen as nn
from flax import struct
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from gen_env.evo.individual import IndividualPlaytraceData
import gymnax
import hydra
import jax
from jax import numpy as jnp
from jax.random import PRNGKey
import numpy as np
import optax
from tensorboardX import SummaryWriter
import tqdm

from gen_env.configs.config import EvoConfig, ILConfig
from gen_env.envs.gen_env import GenEnv
from gen_env.envs.play_env import PlayEnv
from gen_env.utils import init_base_env, init_config
from pcgrl_utils import get_network
from purejaxrl.experimental.s5.wrappers import LogWrapper
from utils import evaluate_on_env_params, get_rand_train_envs, init_il_config, load_elite_envs



Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks'])
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]

# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class Model:
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Optional[Callable[[Params], Any]] = None,
            grads: Optional[Any] = None,
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:
        assert (loss_fn is not None or grads is not None,
                'Either a loss function or grads must be specified.')
        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            assert (has_aux,
                    'When grads are provided, expects no aux outputs.')

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


@partial(jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def _sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == 'det':
        return rng, actor_apply_fn({'params': actor_params}, observations,
                                   temperature)
    else:
        dist = actor_apply_fn({'params': actor_params}, observations,
                              temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)


def sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_apply_fn, actor_params, observations,
                           temperature, distribution)


def log_prob_update(train_state, batch: Batch,
                    rng: PRNGKey) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, val = train_state.apply_fn({'params': actor_params},
                              batch.observations,
                              rngs={'dropout': key}
                            )
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -log_probs.mean()
        return actor_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    total_loss, grads = grad_fn(
        train_state.params,
    )
    # jax.debug.print("total_loss={total_loss}", total_loss=total_loss)
    train_state = train_state.apply_gradients(grads=grads)

    return (rng, train_state, {'actor_loss': total_loss})


def mse_update(train_state: TrainState, batch: Batch,
               rng: PRNGKey) -> Tuple[Model, InfoDict]:
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, val = train_state.apply_fn({'params': actor_params},
                                 batch.observations,
                                #  training=True,
                                 rngs={'dropout': key}
                                )
        actions = dist.sample(seed=key)
        actor_loss = ((actions - batch.actions)**2).mean()
        return actor_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    total_loss, grads = grad_fn(
        train_state.params,
    )
    # jax.debug.print("total_loss={total_loss}", total_loss=total_loss)
    train_state = train_state.apply_gradients(grads=grads)

    return (rng, train_state, {'actor_loss': total_loss})


_log_prob_update_jit = jax.jit(log_prob_update)
_mse_update_jit = jax.jit(mse_update)


class BCLearner(object):

    def __init__(self,
                 cfg: ILConfig,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 1e-3,
                 num_steps: int = int(1e6),
                 hidden_dims: Sequence[int] = (256, 256),
                 distribution: str = 'det'):

        self.distribution = distribution

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)


        schedule_fn = optax.cosine_decay_schedule(-actor_lr, num_steps)
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
            optax.adam(cfg.il_lr, eps=1e-5),
            # optax.scale_by_adam(),
            # optax.scale_by_schedule(schedule_fn)
        )

        env, env_params = init_base_env(cfg)
        env = LogWrapper(env)
        network = get_network(env, env_params, cfg)
        init_x = env.gen_dummy_obs(env_params)
        network_params = network.init(actor_key, init_x)
        print(network.subnet.tabulate(rng, init_x.map, init_x.flat))

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"],
            tx=tx,
        )
        self.train_state = train_state

        # FIXME: load this from checkpoint
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        self.rng, actions = sample_actions(self.rng,
                                                    self.train_state.apply_fn,
                                                    self.train_state.params,
                                                    observations, temperature,
                                                    self.distribution)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

def update(rng: jax.random.PRNGKey, train_state: TrainState, batch: Batch) -> InfoDict:
    # if self.distribution == 'det':
    # self.rng, self.actor, info = _mse_update_jit(
    #     self.train_state, batch, self.rng)
    # else:
    rng, train_state, info = _log_prob_update_jit(
        train_state, batch, rng)
    return rng, train_state, info

        
def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):

    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray,
                 size: int, obs_rew_norm: bool, hide_rules: bool, obs_win: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.size = size

        if obs_win != -1:
            full_win_size = self.observations.map.shape[1]
            assert obs_win < full_win_size
            assert full_win_size == self.observations.map.shape[2]
            lpad = (full_win_size - obs_win) // 2
            rpad = full_win_size - obs_win - lpad
            new_map_obs = jnp.zeros_like(self.observations.map)
            new_map_obs = new_map_obs.at[:, lpad:-rpad, lpad:-rpad].set(
                self.observations.map[:, lpad:-rpad, lpad:-rpad])
            self.observations = self.observations.replace(map=new_map_obs)

        if hide_rules:
            self.observations = self.observations.replace(flat=self.observations.flat.at[:].set(0))

        # FIXME: Terrible HACK.
        # TODO: Make rule_reward_norm a property of the observation dataclass to avoid sketchiness here.
        if not obs_rew_norm:
            self.observations = self.observations.replace(flat=self.observations.flat.at[:, -2:].set(0))

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        observations = jax.tree.map(lambda x: x[indx], self.observations)
        return Batch(observations=observations,
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx])
    
        
class AutoverseILDataset(Dataset):

    def __init__(self,
                 dataset,
                 obs_rew_norm: bool,
                 hide_rules: bool,
                 obs_win: int,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):

            # Dataset leaves have shape (n. eps, n. steps, ...), we flatten these two dimensions.
            dataset = jax.tree.map(lambda x: jnp.concatenate(x, 0), dataset)

            # Remove all entries where the action is -1 (early episode termination due to search iteration cap)
            mask = dataset.action_seq != -1
            dataset = jax.tree.map(lambda x: x[mask], dataset)

            dones_float = dataset.done_seq

            dataset = {
                'observations': dataset.obs_seq,
                'actions': dataset.action_seq,
                'rewards': dataset.rew_seq,
                'terminals': dataset.done_seq,
            }

            super().__init__(dataset['observations'],
                            actions=dataset['actions'].astype(np.float32),
                            rewards=dataset['rewards'].astype(np.float32),
                            masks=1.0 - dataset['terminals'].astype(np.float32),
                            dones_float=dones_float.astype(np.float32),
                            size=len(dataset['rewards']), obs_rew_norm=obs_rew_norm,
                            hide_rules=hide_rules, obs_win=obs_win,
                            )


                         
def evaluate(agent, env: GenEnv, num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats

    
@struct.dataclass
class ILDataset:
    action_seq: chex.Array
    obs_seq: chex.Array
    rew_seq: chex.Array
    done_seq: chex.Array 


from orbax import checkpoint as ocp


def save_checkpoint(ckpt_manager, train_state, t, rng):
    items = {'train_state': train_state, 'rng': rng}
    # save_args = flax.training.orbax_utils.save_args_from_target(items)
    # ckpt_manager.save(t, items=items, save_kwargs={'save_args': save_args})
    ckpt_manager.save(t, args=ocp.args.StandardSave(items))
    ckpt_manager.wait_until_finished() 


@hydra.main(version_base="1.3", config_path="gen_env/configs", config_name="il")
def main(cfg: ILConfig):
    # Try/except to avoid submitit-launcher-plugin swallowing up our error tracebacks.
    try:
        _main(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

    
def init_bc_agent(cfg: ILConfig, env: PlayEnv):
    agent = BCLearner(cfg=cfg, seed=cfg.seed,
                      observations=env.observation_space.sample()[np.newaxis],
                      actions=np.array(env.action_space.sample())[np.newaxis], actor_lr=cfg.il_lr,
                      num_steps=cfg.il_max_steps)
    rng, train_state = agent.rng, agent.train_state

    # FIXME this is silly, lift this out!
    train_state = agent.train_state

    options = ocp.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        cfg._il_ckpt_dir,
        options=options)

    if checkpoint_manager.latest_step() is not None:
        t = checkpoint_manager.latest_step()
        items = {'train_state': train_state, 'rng': rng}
        restore_args = flax.training.orbax_utils.save_args_from_target(items)
        # ckpt = checkpoint_manager.restore(t, items=items, restore_kwargs={'restore_args': restore_args})
        ckpt = checkpoint_manager.restore(t, args=ocp.args.StandardRestore(items))
        train_state, rng = ckpt['train_state'], ckpt['rng']
        # train_state = checkpoint_manager.restore(t, args=ocp.args.StandardRestore(agent.train_state))
        checkpoint_manager.wait_until_finished()
        # agent.train_state = train_state
    else: 
        t = 0

    return rng, train_state, t, checkpoint_manager


def log_callback(cfg, summary_writer, update_info, i):
    for k, v in update_info.items():
        print(f'{k}: {v}')
        if isinstance(v, jnp.ndarray):
            assert v.shape == ()
            v = v.item()
        summary_writer.add_scalar(f'il/{k}', v, i)
    # summary_writer.add_scalar('il/fps', cfg.il_batch_size * i / (timer() - start_time), i)


def _main(cfg: ILConfig):
    init_config(cfg)
    latest_gen = init_il_config(cfg)
    assert latest_gen is not None, \
            "Must select an evo-gen from which to load playtraces for imitation learning when running IL script." +\
            "Set `load_gen=-1` to load latest generation for which playtraces have been aggregated."

    # Environment class doesn't matter and will be overwritten when we load in an individual.
    # env = maze.make_env(10, 10)
    env, env_params = init_base_env(cfg)
    # rng = np.random.default_rng(cfg.env_exp_id)
    rng = jax.random.PRNGKey(cfg.seed)

    if cfg.overwrite:
        if os.path.exists(cfg._log_dir_il):
            shutil.rmtree(cfg._log_dir_il)
            print(f"Overwriting {cfg._log_dir_il}")

    if not os.path.exists(cfg._log_dir_il):
        os.makedirs(cfg._log_dir_il)


    # Load the transitions from the training set
    train_elites, val_elites = load_elite_envs(cfg._log_dir_common, latest_gen)

    # Initialize tensorboard logger
    summary_writer = SummaryWriter(
        os.path.join(cfg._log_dir_il, 'tb', str(cfg.seed)))

    _log_callback = partial(log_callback, cfg=cfg, summary_writer=summary_writer)

    def log_callback_main(cfg, summary_writer, update_info, i, start_time, t):
        update_info.update({'fps': cfg.il_batch_size * (i + 1 - t) / (timer() - start_time)})
        # log_callback(cfg, summary_writer, update_info, i)
        jax.experimental.io_callback(_log_callback, None, update_info=update_info, i=i)

    def eval_pct_actions_correct(ds, n, rng, train_state, i, summary_writer, cfg):
        batch = ds.sample(cfg.il_batch_size)

        rng, _ = jax.random.split(rng)
        dist, val = train_state.apply_fn({'params': train_state.params},
                            batch.observations,
                            rngs={'dropout': rng}
        )
        actions = dist.sample(seed=rng)
        pct_correct = (actions == batch.actions).mean()
        jax.experimental.io_callback(_log_callback, None,
            update_info={f'{n}_pct_correct': pct_correct}, i=i,
        )
        # summary_writer.add_scalar(f'il/eval/{n}_pct_correct', pct_correct, i)
        # print(f"pct. {n} correct: {pct_correct}")


    _datasets = (train_elites, val_elites)
    datasets = []
    for d in _datasets:
        d = AutoverseILDataset(dataset=ILDataset(
            action_seq=d.action_seq,
            obs_seq=d.obs_seq,
            rew_seq=d.rew_seq,
            done_seq=d.done_seq
        ), 
        obs_rew_norm=cfg.obs_rew_norm,
        hide_rules=cfg.hide_rules,
        obs_win=cfg.obs_window,
    )
        datasets.append(d)
    train_dataset, val_dataset = datasets

    video_save_folder = None if cfg.render_freq == -1 else os.path.join(
        cfg._log_dir_il, 'video', 'eval')


    kwargs = cfg.__dict__
    kwargs['num_steps'] = cfg.il_max_steps
    rng, train_state, t, checkpoint_manager = init_bc_agent(cfg, env)

    _log_callback_fps = partial(log_callback_main, cfg=cfg, summary_writer=summary_writer, t=t)
    _save_checkpoint = partial(save_checkpoint, ckpt_manager=checkpoint_manager)

    # Prepare the evaluation functions
    _evaluate_on_env_params = partial(
            evaluate_on_env_params, mode="il", n_eps=1, env=env, cfg=cfg,
            writer=summary_writer, network_apply_fn=train_state.apply_fn)
    _evaluate_on_env_params_train = jax.jit(partial(
            _evaluate_on_env_params, params_type="train", env_params=train_elites.env_params,
            search_rewards=train_elites.rew_seq))
    _evaluate_on_env_params_val = jax.jit(partial(
            _evaluate_on_env_params, params_type="val", env_params=val_elites.env_params,
            search_rewards=val_elites.rew_seq))

    for n, elites in zip(['train', 'val'], [train_elites, val_elites]):
        elites: IndividualPlaytraceData
        search_rewards = elites.rew_seq.sum(axis=1)
        search_rewards = (search_rewards + elites.env_params.rew_bias) * elites.env_params.rew_scale
        metric = {
            'mean_return_search': search_rewards.mean(),
            'max_return_search': search_rewards.max(),
            'min_return_search': search_rewards.min(),
        }
        for k, v in metric.items():
            name = f'il/eval/{n}_{k}'
            summary_writer.add_scalar(name, v, t)

    def eval_callback(train_state, i, rng, cfg, train_dataset, val_dataset, summary_writer):

        for ds, n in zip([train_dataset, val_dataset], ['train', 'val']):
            eval_pct_actions_correct(ds, n, rng, train_state, i, summary_writer, cfg)

        _evaluate_on_env_params_train(rng=rng, 
                            network_params={'params':train_state.params}, update_i=i,
                            )
        _evaluate_on_env_params_val(rng=rng,
                            network_params={'params': train_state.params}, update_i=i,
                            )

    _eval_callback = partial(eval_callback, cfg=cfg, train_dataset=train_dataset, val_dataset=val_dataset,
                             summary_writer=summary_writer)

    rng, _ = jax.random.split(rng)

    # Just evaluate the agent once then quit
    if cfg.evaluate:
        _evaluate_on_env_params(rng, cfg, env, train_elites.env_params, train_state.apply_fn,
                               {'params': train_state.params}, t,
                               summary_writer, n_eps=10, mode="il", params_type="train")
        _evaluate_on_env_params(rng, cfg, env, val_elites.env_params, train_state.apply_fn,
                               {'params': train_state.params}, t,
                               summary_writer, n_eps=10, mode="il", params_type="val")
        return

    eval_returns = []
    start_time = timer()
    # for i in tqdm.tqdm(range(t, cfg.il_max_steps + 1),
    #                    smoothing=0.1,
    #                    disable=not cfg.il_tqdm):

    def train_step(carry, i):
        rng, train_state = carry
        batch = train_dataset.sample(cfg.il_batch_size)

        rng, train_state, update_info = _log_prob_update_jit(batch=batch, train_state=train_state, rng=rng)

        do_log = i % cfg.log_interval == 0
        jax.lax.cond(do_log, lambda: _log_callback_fps(update_info=update_info, i=i, start_time=start_time), lambda: None)

        do_eval = (cfg.eval_interval != -1) and (i % cfg.eval_interval == 0)
        jax.lax.cond(do_eval,
            lambda: _eval_callback(train_state=train_state, i=i, rng=rng), lambda: None,)


            # eval_stats = evaluate(agent, env, cfg.eval_episodes)

            # for k, v in eval_stats.items():
            #     summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            # summary_writer.flush()

            # eval_returns.append((i, eval_stats['return']))
            # np.savetxt(os.path.join(cfg._log_dir_il, f'{cfg.seed}.txt'),
            #            eval_returns,
            #            fmt=['%d', '%.1f'])
                       
        do_ckpt = cfg.ckpt_interval != -1 and i % cfg.ckpt_interval == 0
        jax.lax.cond(do_ckpt, lambda: jax.experimental.io_callback(_save_checkpoint, None, train_state=train_state, t=i,
        rng=rng), lambda: None)

        return (rng, train_state), ()

    steps_remaining = cfg.il_max_steps + 1 - t
    jax.lax.scan(train_step, (rng, train_state), jnp.arange(t, cfg.il_max_steps+1), length=steps_remaining)

if __name__ == "__main__":
    main()