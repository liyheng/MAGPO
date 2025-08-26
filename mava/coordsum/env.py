from functools import cached_property, partial
from typing import Tuple, TYPE_CHECKING, NamedTuple
import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition
if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass



@dataclass
class State:
    """A dataclass representing the state of the game.
    step_count: an integer representing the current step of the episode.
    key: a pseudorandom number generator key.
    """

    step_count: chex.Array  # ()
    target: chex.Array  # ()
    record: chex.Array  # (num_step, action)
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """The observation that the agent sees.
    agent_obs: the agent's observation.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_obs: chex.Array  # (1,)
    step_count: chex.Array  # ()


class CoordSum(Environment[State, specs.MultiDiscreteArray, Observation]):
    def __init__(
        self,
        num_agents: int,
        num_actions: int,
        time_limit: int = 100,
        maxval: int = None,
    ):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.time_limit = time_limit
        if maxval:
            self.maxval = maxval
        else:
            self.maxval = num_actions

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        key, target_key = jax.random.split(key)
        target = jax.random.randint(target_key, (self.time_limit+1,), minval=0, maxval=self.maxval)
        record = -jnp.ones((self.num_actions, self.time_limit), dtype=jnp.int32)
        state = State(
            step_count=jnp.array(0, int),
            target=target,
            record=record,
            key=key,
        )

        # collect first observations and create timestep
        agent_obs = jnp.array([[target[0]]]*self.num_agents, jnp.int32)
        
        observation = Observation(
            agent_obs=agent_obs,
            step_count=state.step_count,
        )
        timestep = restart(observation=observation, shape=self.num_agents)
        return state, timestep

    def step(
        self,
        state: State,
        actions: chex.Array,
    ) -> Tuple[State, TimeStep[Observation]]:

        target_t = state.target[state.step_count]  # 当前 step 的 target 值
        sum_match = jnp.sum(actions) == target_t

        record_row = state.record[target_t]  # shape: (T,), T=100

        # mask: 1 for valid entries, 0 for -1
        mask = (record_row != -1).astype(jnp.float32)

        safe_entries = jnp.where(mask == 1.0, record_row, 0)

        counts = jnp.bincount(
            safe_entries.astype(jnp.int32),   # shape static, dtype int32
            weights=mask,                     # shape static
            minlength=self.num_actions,        # static
            length=self.time_limit
        )

        guess = jnp.argmax(counts)

        hit = guess == actions[0]
        reward = jnp.where(sum_match, jnp.where(hit, 1.0, 2.0), 0.0)
        rewards = jnp.full((self.num_agents,), reward)
      
        new_record = jax.lax.dynamic_update_slice(
            state.record,
            jnp.array([[actions[0]]]),
            (target_t, state.step_count)  # slice starts at (row, col)
        )
        # construct timestep and check environment termination
        steps = state.step_count + 1
        done = steps >= self.time_limit

        # compute next observation
        agent_obs = jnp.array([[state.target[steps]]]*self.num_agents, jnp.int32)

        next_observation = Observation(
            agent_obs=agent_obs,
            step_count=steps,
        )

        timestep = jax.lax.cond(
            done,
            lambda: termination(
                reward=rewards, observation=next_observation, shape=self.num_agents
            ),
            lambda: transition(
                reward=rewards, observation=next_observation, shape=self.num_agents
            ),
        )

        # create environment state
        next_state = State(
            step_count=steps,
            target=state.target,
            record=new_record,
            key=state.key,
        )
        return next_state, timestep

    def _make_agent_observation(
        self,
        state: State,
        agent_ID: int,
    ) -> chex.Array:
        
        return jnp.array(state.target[state.step_count], jnp.int32)

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the MatrixGame environment.
        Returns:
            Spec for the `Observation`, consisting of the fields:
                - agent_obs: BoundedArray (int32) of shape (num_agents, num_agents).
                - step_count: BoundedArray (int32) of shape ().
        """

        obs_shape = (self.num_agents,1)
        low = jnp.zeros(obs_shape)
        high = jnp.ones(obs_shape) * self.maxval

        agent_obs = specs.BoundedArray(obs_shape, jnp.int32, low, high, "agent_obs")
        step_count = specs.BoundedArray((), jnp.int32, 0, self.time_limit, "step_count")
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_obs=agent_obs,
            step_count=step_count,
        )

    @cached_property
    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec.
        Since this is a multi-agent environment, the environment expects an array of actions.
        This array is of shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([self.num_actions] * self.num_agents, jnp.int32),
            name="action",
        )
    