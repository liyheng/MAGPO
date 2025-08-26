import copy
from functools import partial
import time
from typing import Any, Tuple, Callable

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import get_eval_fn, get_num_eval_envs
from mava.networks import RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks import ScannedRNN
from mava.systems.ppo.types import (
    HiddenStates,
    OptStates,
    Params,
    RNNLearnerState,
    RNNPPOTransition,
)
from mava.types import (
    ExperimentOutput,
    LearnerFn,
    MarlEnv,
    Metrics,
    RecActorApply,
    RecCriticApply,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics

def move_agent_to_front(x):
    return jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim)))

def batched_actor_apply_fn(
    actor_apply_fn: Callable,
    params: FrozenDict,                            # [num_agents, ...]
    h_state: jnp.ndarray,                                # [B, A, H]
    obs_done: Tuple[jnp.ndarray, jnp.ndarray],
    key: chex.PRNGKey
):
    """Apply actor for all agents in batch via vmap over agent axis (A)."""
    obs, done = obs_done

    # Transpose [B, A, ...] -> [A, B, ...] to vmap over agent
    h_state = jnp.swapaxes(h_state, 0, 1) [:, :, None, ...]       # [A, B, H]
    done = jnp.transpose(done, (2, 0, 1))[:, :, :, None]   
    obs = tree.map(move_agent_to_front, obs)
    obs = tree.map(lambda x: x[:, :, :, None, ...], obs)           # [A, 1, B, 1, D]

    # Define single-agent batched apply (over B)
    def _apply_single_agent(params, h_i, obs_i, done_i, key_i):
        h_new, policy = actor_apply_fn(params, h_i, (obs_i, done_i))
        action = policy.sample(seed=key_i)
        log_prob = policy.log_prob(action)
        return h_new, action, log_prob
    
    h_new, action, log_prob = jax.vmap(_apply_single_agent)(params, h_state, obs, done, key)

    return h_new, action, log_prob

def batched_eval_fn(
    actor_apply_fn: Callable,
    params: FrozenDict,                            # [num_agents, ...]
    h_state: jnp.ndarray,                                # [B, A, H]
    obs_done: Tuple[jnp.ndarray, jnp.ndarray],
    config: DictConfig,
    key: chex.PRNGKey,
):
    """Apply actor for all agents in batch via vmap over agent axis (A)."""
    obs, done = obs_done

    # Transpose [B, A, ...] -> [A, B, ...] to vmap over agent
    h_state = jnp.swapaxes(h_state, 0, 1) [:, :, None, ...]       # [A, B, H]
    done = jnp.transpose(done, (2, 0, 1))[:, :, :, None]   
    obs = tree.map(move_agent_to_front, obs)
    obs = tree.map(lambda x: x[:, :, :, None, ...], obs)           # [A, 1, B, 1, D]

    # Define single-agent batched apply (over B)
    def _apply_single_agent(params, h_i, obs_i, done_i, key_i):
        h_new, pi = actor_apply_fn(params, h_i, (obs_i, done_i))
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key_i)
        return h_new, action
    
    h_new, action = jax.vmap(_apply_single_agent)(params, h_state, obs, done, key)

    return h_new, action

def batched_value_apply_fn(
    actor_apply_fn: Callable,
    params: FrozenDict,                            # [num_agents, ...]
    h_state: jnp.ndarray,                                # [B, A, H]
    obs_done: Tuple[jnp.ndarray, jnp.ndarray] # obs: [B, A, ...], done: [B, A]
):
    """Apply actor for all agents in batch via vmap over agent axis (A)."""
    obs, done = obs_done

    # Transpose [B, A, ...] -> [A, B, ...] to vmap over agent
    h_state = jnp.swapaxes(h_state, 0, 1) [:, :, None, ...]       # [A, B, H]
    done = jnp.transpose(done, (2, 0, 1))[:, :, :, None]   
    obs = tree.map(move_agent_to_front, obs)
    obs = tree.map(lambda x: x[:, :, :, None, ...], obs)           # [A, 1, B, 1, D]

    # Define single-agent batched apply (over B)
    def _apply_single_agent(params, h_i, obs_i, done_i):
        return actor_apply_fn(params, h_i, (obs_i, done_i))  # already batched over B

    return  jax.vmap(_apply_single_agent)(params, h_state, obs, done)

def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[RecActorApply, RecCriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[RNNLearnerState]:
    """Get the learner function."""
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns
    num_agents = env.num_agents

    def _update_step(learner_state: RNNLearnerState, _: Any) -> Tuple[RNNLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - last_done (bool): Whether the last timestep was a terminal state.
                - hstates (HiddenStates): The hidden state of the policy and critic RNN.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, Tuple[RNNPPOTransition, Metrics]]:
            """Step the environment."""
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                last_done,
                last_hstates,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            # Add a batch dimension to the observation.
            batched_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
            ac_in = (batched_observation, last_done[jnp.newaxis, :])
            # Run the network.
            policy_key = jax.random.split(policy_key, num_agents)
            policy_hidden_state, actions, log_probs = batched_actor_apply_fn(
                actor_apply_fn, params.actor_params, last_hstates.policy_hidden_state, ac_in, policy_key
            )
            critic_hidden_state, value = batched_value_apply_fn(
                critic_apply_fn, params.critic_params, last_hstates.critic_hidden_state, ac_in
            )
            policy_hidden_state = jnp.swapaxes(policy_hidden_state, 0, 1).squeeze(2)
            critic_hidden_state = jnp.swapaxes(critic_hidden_state, 0, 1).squeeze(2)
            action = jnp.swapaxes(actions, 0, 3).squeeze(0).squeeze(0)
            log_prob = jnp.swapaxes(log_probs, 0, 3).squeeze(0).squeeze(0)
            value = jnp.swapaxes(value, 0, 3).squeeze(0).squeeze(0)

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            hstates = HiddenStates(policy_hidden_state, critic_hidden_state)
            transition = RNNPPOTransition(
                last_done,
                action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                last_hstates,
            )
            learner_state = RNNLearnerState(
                params, opt_states, key, env_state, timestep, done, hstates
            )
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            return learner_state, (transition, metrics)

        # Step environment for rollout length
        learner_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # Calculate advantage
        params, opt_states, key, env_state, last_timestep, last_done, hstates = learner_state

        # Add a batch dimension to the observation.
        batched_last_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
        ac_in = (batched_last_observation, last_done[jnp.newaxis, :])
        # Run the network.
        _, last_val = batched_value_apply_fn(
                critic_apply_fn, params.critic_params, hstates.critic_hidden_state, ac_in
            )
        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = jnp.swapaxes(last_val, 0, 3).squeeze(0).squeeze(0)

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_states, agent_perm, key = train_state
                traj_batch, advantages, targets = batch_info
                def _update_single_agent(carry, agent_idx):
                    agent_idx = agent_perm[agent_idx]
                    params, opt_states, adv_weight, key = carry

                    actor_param = tree.map(lambda p: p[agent_idx], params.actor_params)
                    actor_opt_state = tree.map(lambda p: p[agent_idx], opt_states.actor_opt_state)
                    critic_param = tree.map(lambda p: p[agent_idx], params.critic_params)
                    critic_opt_state = tree.map(lambda p: p[agent_idx], opt_states.critic_opt_state)
                    gae = advantages[..., agent_idx,None]
                    agent_target = targets[..., agent_idx,None]
                    agent_value = traj_batch.value[..., agent_idx,None]
                    agent_old_log_prob = traj_batch.log_prob[..., agent_idx,None]
                    agent_action = traj_batch.action[:,:, agent_idx,None]
                    obs_for_agent = tree.map(lambda x: x[:, :, agent_idx, None, ...], traj_batch.obs)
                    done_for_agent = traj_batch.done[:, :, agent_idx, None]
                    obs_and_done = (obs_for_agent, done_for_agent)
                    h_state = traj_batch.hstates.policy_hidden_state[0][:,agent_idx,None]
                    h_state_c = traj_batch.hstates.critic_hidden_state[0][:,agent_idx,None]
                    def _actor_loss_fn(actor_params, gae, agent_old_log_prob, agent_action, obs_and_done, h_state, adv_weight, key):
                        _, actor_policy = actor_apply_fn(
                            actor_params, h_state, obs_and_done
                        )
                        log_prob = actor_policy.log_prob(agent_action)
                        ratio = jnp.exp(log_prob - agent_old_log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        m_adv = gae * adv_weight
                        actor_loss1 = ratio * m_adv
                        actor_loss2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.system.clip_eps,
                                1.0 + config.system.clip_eps,
                            )
                            * m_adv
                        )
                        actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                        actor_loss = actor_loss.mean()
                        # The seed will be used in the TanhTransformedDistribution:
                        entropy = actor_policy.entropy(seed=key).mean()
                        total_loss = actor_loss - config.system.ent_coef * entropy
                        return total_loss, (actor_loss, entropy)

                    def _critic_loss_fn(
                        critic_params: FrozenDict,
                        obs_and_done,
                        h_state,
                        agent_value,
                        targets: chex.Array,
                    ) -> Tuple:
                        """Calculate the critic loss."""
                        # Rerun network
                        _, value = critic_apply_fn(
                            critic_params, 
                            h_state, 
                            obs_and_done
                        )

                        # Clipped MSE loss
                        value_pred_clipped = agent_value + (value - agent_value).clip(
                            -config.system.clip_eps, config.system.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        total_loss = config.system.vf_coef * value_loss
                        return total_loss, value_loss

                    # Calculate critic loss

                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    key, entropy_key = jax.random.split(key)
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)

                    value_loss_info, critic_grads = critic_grad_fn(
                        critic_param, obs_and_done, h_state_c, agent_value, agent_target
                    )
                    actor_loss_info, actor_grads = actor_grad_fn(
                        actor_param, gae, agent_old_log_prob, agent_action, obs_and_done, h_state, adv_weight, entropy_key
                    )
                    # Compute the parallel mean (pmean) over the batch.
                    # This pmean could be a regular mean as the batch axis is on the same device.
                    actor_grads, actor_loss_info = jax.lax.pmean((actor_grads, actor_loss_info), axis_name="batch")
                    # pmean over devices.
                    actor_grads, actor_loss_info = jax.lax.pmean((actor_grads, actor_loss_info), axis_name="device")
                    critic_grads, value_loss_info = jax.lax.pmean((critic_grads, value_loss_info), axis_name="batch")
                    # pmean over devices.
                    critic_grads, value_loss_info = jax.lax.pmean((critic_grads, value_loss_info), axis_name="device")
                    # Update params and optimiser state
                    actor_updates, actor_new_opt_state = actor_update_fn(
                        actor_grads, actor_opt_state
                    )
                    actor_new_param = optax.apply_updates(
                        actor_param, actor_updates
                    )
                    critic_updates, critic_new_opt_state = critic_update_fn(
                        critic_grads, critic_opt_state
                    )
                    critic_new_param = optax.apply_updates(
                        critic_param, critic_updates
                    ) 
                    # after-calculate
                    _, actor_policy = actor_apply_fn(
                            actor_new_param, h_state, obs_and_done
                        )
                    new_log_prob = actor_policy.log_prob(agent_action)
                    ratio = jnp.exp(new_log_prob - agent_old_log_prob)
                    new_weight = adv_weight * jax.lax.stop_gradient(ratio)
                    new_actor_params = tree.map(lambda p, new_p: p.at[agent_idx].set(new_p),
                                                params.actor_params,
                                                actor_new_param)
                    new_actor_opt_states = tree.map(lambda p, new_p: p.at[agent_idx].set(new_p),
                                                opt_states.actor_opt_state,
                                                actor_new_opt_state)
                    new_critic_params = tree.map(lambda p, new_p: p.at[agent_idx].set(new_p),
                                                params.critic_params,
                                                critic_new_param)
                    new_critic_opt_states = tree.map(lambda p, new_p: p.at[agent_idx].set(new_p),
                                                opt_states.critic_opt_state,
                                                critic_new_opt_state)
                    new_params = Params(new_actor_params, new_critic_params)
                    new_opt_state = OptStates(new_actor_opt_states, new_critic_opt_states)

                    carry_out = (new_params, new_opt_state, new_weight, key)

                    value_loss, unscaled_value_loss = value_loss_info
                    actor_loss, (_, entropy) = actor_loss_info
                    total_loss = actor_loss + value_loss
                    loss_info = {
                        "total_loss": total_loss,
                        "value_loss": unscaled_value_loss,
                        "actor_loss": actor_loss,
                        "entropy": entropy,
                    }
                    
                    return carry_out, loss_info
                
                carry_init = (params, opt_states, jnp.ones_like(advantages[..., 0, None]), key)
                carry_out, loss_info = jax.lax.scan(
                    _update_single_agent,
                    carry_init,
                    jnp.arange(config.system.num_agents)
                )
                new_params, new_opt_state, _, key = carry_out

                return (new_params, new_opt_state, agent_perm, key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, agent_shuffle_key, entropy_key = jax.random.split(key, 4)
            batch = (traj_batch, advantages, targets)

            # # Shuffle agents
            agent_perm = jax.random.permutation(agent_shuffle_key, config.system.num_agents)
            #batch = tree.map(lambda x: jnp.take(x, agent_perm, axis=2), batch)

            # Shuffle minibatches
            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )
            batch = tree.map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
            )
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
            reshaped_batch = tree.map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = tree.map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

            # Update minibatches
            (params, opt_states, agent_perm, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, agent_perm, entropy_key), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
        )
        return learner_state, (episode_metrics, loss_info)

    def learner_fn(learner_state: RNNLearnerState) -> ExperimentOutput[RNNLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstates (HiddenStates): The hidden state of the policy and critic RNN.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RNNLearnerState], Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    num_agents = env.num_agents
    config.system.num_agents = num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)

    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        hidden_state_dim=config.network.hidden_state_dim,
    )
    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        centralised_critic=True,
    )

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation with obs of all agents.
    init_obs = env.observation_spec.generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)

    # Initialise hidden state.
    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    actor_keys = jax.random.split(actor_net_key, num_agents)
    critic_keys = jax.random.split(critic_net_key, num_agents)
    # initialise params and optimiser state.

    policy_hs_array = jnp.swapaxes(init_policy_hstate, 0, 1)[:, :, None, ...] 
    critic_hs_array = jnp.swapaxes(init_critic_hstate, 0, 1)[:, :, None, ...] 


    obs_for_vmap = tree.map(move_agent_to_front, init_obs)
    done_for_vmap = jnp.transpose(init_done, (2, 0, 1))                         
    # Step 2: expand axis=3 to preserve A=1 → [A, 1, B, 1, D]
    obs_for_vmap = tree.map(lambda x: x[:, :, :, None, ...], obs_for_vmap)           # [A, 1, B, 1, D]
    done_for_vmap = done_for_vmap[:, :, :, None]  
    actor_params = jax.vmap(actor_network.init)(actor_keys, policy_hs_array, (obs_for_vmap, done_for_vmap))
    critic_params = jax.vmap(critic_network.init)(critic_keys, critic_hs_array, (obs_for_vmap, done_for_vmap))
    actor_opt_state = jax.vmap(actor_optim.init)(actor_params)
    critic_opt_state = jax.vmap(critic_optim.init)(critic_params)
    
    # Get network apply functions and optimiser updates.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Pack params and initial states.
    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        # Update the params and hstates
        params = restored_params
        hstates = restored_hstates if restored_hstates else hstates

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, hstates, step_keys, dones)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, hstates, step_keys, dones = replicate_learner
    init_learner_state = RNNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        dones=dones,
        hstates=hstates,
    )
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "rec_happo"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert (
            config.system.rollout_length % config.system.recurrent_chunk_size == 0
        ), "Rollout length must be divisible by recurrent chunk size."

        assert (
            config.arch.num_envs % config.system.num_minibatches == 0
        ), "Number of envs must be divisibile by number of minibatches."

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config=config, add_global_state=True)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )
    def make_rec_eval_act_fn(actor_apply_fn: RecActorApply, config: DictConfig):
        """Makes an act function that conforms to the evaluator API given a standard
        recurrent mava actor network."""
        _hidden_state = "hidden_state"
        def eval_act_fn(
            params, timestep, key, actor_state
        ):
            hidden_state = actor_state[_hidden_state]
            n_agents = timestep.observation.agents_view.shape[1]
            last_done = timestep.last()[:, jnp.newaxis].repeat(n_agents, axis=-1)
            ac_in = (timestep.observation, last_done)
            ac_in = tree.map(lambda x: x[jnp.newaxis], ac_in)  # add batch dim to obs
            key = jax.random.split(key, config.system.num_agents)
            hidden_state, action = batched_eval_fn(actor_apply_fn, 
                                                   params, 
                                                   hidden_state, 
                                                   ac_in,
                                                   config,
                                                   key)
            hidden_state = jnp.swapaxes(hidden_state, 0, 1).squeeze(2)
            action = jnp.swapaxes(action, 0, 3).squeeze(0)
            return action.squeeze(0), {_hidden_state: hidden_state}

        return eval_act_fn
    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_rec_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )
    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    eval_hs = ScannedRNN.initialize_carry(
        (n_devices, eval_batch_size, config.system.num_agents),
        config.network.hidden_state_dim,
    )
    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)

        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        #trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        trained_params = tree.map(lambda x: x[:, 0, ...], learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {"hidden_state": eval_hs})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])


        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        eval_hs = ScannedRNN.initialize_carry(
            (n_devices, eval_batch_size, config.system.num_agents),
            config.network.hidden_state_dim,
        )
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": eval_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_happo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent MAPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
