import jax 
from jax import numpy as jnp
import chex
import optax
from functools import partial
from typing import Any, Callable

from flax import linen as nn
import flax
from flax.training.train_state import TrainState

from buffer import Transition


class DQNTrainingArgs:
    gamma: float = 0.99 # discounting factor in MDP
    learning_rate: float = 2.5e-4 # learning rate for DQN parameter optimization
    target_update_every: int = 512 # the target network update frequency (per training steps)
    fifo_buffer_size: int = 10000 # the total size of the replay buffer
    buffer_prefill: int = 10000 # the number of transitions to prefill the replay buffer with.
    train_batch_size: int = 128 # the batch size used in training
    start_eps: float = 1.0 # epsilon (of epsilon-greedy action selection) in the beginning of the training
    end_eps: float = 0.05 # epsilon (of epsilon-greedy action selection) in the end of the training
    epsilon_decay_steps: int = 25_000 # how many steps to decay epsilon over
    sample_budget: int = 250_000 # the total number of environment transitions to train our agent over
    eval_env_steps: int = 5000 # total number of env steps to evaluate the agent over
    eval_environments: int = 10 # how many parallel environments to use in evaluation
    # say we do 1 training step per N "environment steps" (i.e. per N sampled MDP transitions); 
    # also, say train batch size in this step is M (in the number of MDP transitions).
    # train_intensity is the desired fraction M/N.
    # i.e. the ratio of "replayed" transitions to sampled transitions
    # the higher this number is, the more intense experience replay will be.
    # to keep the implementation simple, we don't allow to make this number
    # bigger that the batch size but it can be an arbitrarily small positive number
    train_intensity: float = 8.0

class DQN(nn.Module):
    n_actions: int
    state_shape: list[int]
    
    @nn.compact
    def __call__(self, state: '[batch, *state_shape]') -> '[batch, n_actions]':
        """ This function defines the forward pass of Deep Q-Network.
    
        Note that the expected format of convolutional layers is [B, H, W, C]
        Where B - batch dimension, H, W - height and width dimensions respectively
        C - channels dimension
    
        Args:
            state: dtype float32, shape [batch, *state_shape] a batch of states of MDP
        Returns:
            array containing Q-values for each action, its shape is [batch, n_actions]
        """
        batch = state.shape[0]
                
        x = nn.Dense(128)(state)
        x = nn.relu(x)
        
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        q_values = jnp.zeros((batch, self.n_actions), dtype=jnp.float32)

        return q_values


DQNParameters = flax.core.frozen_dict.FrozenDict


class DQNTrainState(TrainState): 
    # Note that `apply_fn`, `params`, and `tx` are inherited from TrainState 
    target_params: DQNParameters


@chex.dataclass(frozen=True)
class DQNAgent:
    dqn: DQN # the Deep Q-Network instance of the agent
    initialize_agent_state: Callable[[Any], DQNTrainState]
    """initialize_agent_state:
    creates the training state for our DQN agent.
    """
    select_action: Callable[[DQN, chex.PRNGKey, DQNParameters, chex.Array, chex.Array], chex.Array]
    """select_action:
    This function takes a random key of jax, a Deep Q-Network instance and its parameters
    as well as the state of MDP and the epsilon parameter and performs the action selection
    with an epsilon greedy strategy. Note that this function should be vmap-able.
    """
    compute_loss: Callable[[DQN, DQNParameters, DQNParameters, Transition, float], chex.Array]
    """compute_loss:
    This function computes the Deep Q-Network loss. It takes as an input the DQN object,
    the current parameters of the DQN agent and target parameters of the 
    DQN agent. Additionally it accepts the `Transition` object (see buffer.py for definition) and
    the gamma discounting factor. 
    """
    update_target: Callable[[DQNTrainState], DQNTrainState]
    """update_target: 
    performs the target network parameters update making the latter equal to the current parameters.
    """


def select_action(dqn: DQN, rng: chex.PRNGKey, params: DQNParameters, state: chex.Array, epsilon: chex.Array) -> chex.Array:
    random_value = jax.random.uniform(rng)
    
    random_action = jax.random.randint(rng, shape=(), minval=0, maxval=dqn.n_actions)
    
    q_values = dqn.apply(params, state)  # Forward pass to get Q-values
    greedy_action = jnp.argmax(q_values, axis=-1)
    
    action = jnp.where(random_value < epsilon, random_action, greedy_action)
    return jnp.array(action, dtype=jnp.int32)



def compute_loss(dqn: DQN, params: DQNParameters, target_params: DQNParameters, transition: Transition, gamma: float) -> chex.Array:
    state, action, reward, done, next_state = transition
    
    q_values = dqn.apply(params, state)  
    
    next_q_values = dqn.apply(target_params, next_state) 
    
    q_value = q_values[action] 
    
    max_next_q_value = jnp.max(next_q_values)  

    target_value = reward + gamma * (1 - done) * max_next_q_value  
    
    loss = jnp.square(q_value - target_value) 
    
    return loss


def update_target(state: DQNTrainState) -> DQNTrainState:
    state = state.replace(target_params = state.params)
    new_state = state
    return new_state


def initialize_agent_state(dqn: DQN, rng: chex.PRNGKey, args: DQNTrainingArgs) -> DQNTrainState:
    
    if not hasattr(args, "state_shape"):
        args.state_shape = (4,)  # Assuming CartPole environment; adjust as needed

    ini_state = jnp.zeros((1, *args.state_shape), dtype=jnp.float32)
    params = dqn.init(rng, ini_state)

    optimizer = optax.adam(learning_rate=args.learning_rate)

    train_state = DQNTrainState.create(
        apply_fn=dqn.apply,
        params=params,
        target_params=params,  
        tx=optimizer,
    )

    return train_state

dqn = DQN(n_actions=2, state_shape=(4,))
SimpleDQNAgent = DQNAgent(
    dqn=dqn,
    initialize_agent_state=initialize_agent_state,
    select_action=select_action,
    compute_loss=compute_loss,
    update_target=update_target,
)


def compute_loss_double_dqn(dqn: DQN, params: DQNParameters, target_params: DQNParameters, transition: Transition, gamma: float) -> chex.Array:
    state, action, reward, done, next_state = transition
    
    q_values = dqn.apply(params, state)
    q_value = jnp.take_along_axis(q_values, action, axis=-1).squeeze()
    
    next_q_values = dqn.apply(params, next_state)
    best_next_action = jnp.argmax(next_q_values, axis=-1)
    
    target_q_values = dqn.apply(target_params, next_state)
    target_q_value = jnp.take_along_axis(target_q_values, best_next_action[..., None], axis=-1).squeeze()
    
    td_target = reward + gamma * target_q_value * (1.0 - done)
    
    loss = jnp.square(td_target - q_value).mean()
    return loss


DoubleDQNAgent = DQNAgent(
    dqn=dqn,
    initialize_agent_state=initialize_agent_state,
    select_action=select_action,
    compute_loss=compute_loss_double_dqn,
    update_target=update_target,
)