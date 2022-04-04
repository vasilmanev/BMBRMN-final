import pickle
from collections import namedtuple, deque
from typing import List

import events as e
from .StateToFeat import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


# Events


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.model.train_bot(old_game_state, self_action, new_game_state, reward_from_events(self, events))

    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 3.5,
        e.KILLED_OPPONENT: 2.5,
        e.KILLED_SELF: -15,
        e.GOT_KILLED: 0, 
        e.CRATE_DESTROYED: 4.5,
        e.BOMB_DROPPED: 1,
        e.MOVED_DOWN: 10,
        e.MOVED_UP: 10,
        e.MOVED_RIGHT: 10,
        e.MOVED_LEFT: 10,
        e.INVALID_ACTION: -20,
        e.WAITED: -5,

        e.SURVIVED_ROUND: 7,
    }
    reward_sum = 0
    last_event = e.WAITED
    print(events)
    for i, event in enumerate(events):
        if event in game_rewards:
            reward_sum += game_rewards[event]
            if event == e.INVALID_ACTION:
                print('invalid')
            if event == e.WAITED:
                print('wait')
        # if event == e.WAITED and last_event == e.BOMB_DROPPED:
        #     reward -= 10
        for j in range(max(0, i-5), i):
            if events[j] == e.BOMB_DROPPED and event in [e.INVALID_ACTION, e.WAITED, e.KILLED_SELF]:
                reward_sum -= 100

        allwaited = True
        for j in range(max(0, i-3), i):
            if events[j] not in [e.WAITED]:
                allwaited = False
        if allwaited:
            # print('ohmaiko')
            reward_sum -= 1


    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # breakpoint()
    return reward_sum
