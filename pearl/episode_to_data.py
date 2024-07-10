from rlgym_sim.utils.gamestates import GameState
import numpy as np
from enum import Enum

from rlgym_sim.utils.math import quat_to_rot_mtx

from pearl.data import EpisodeData

DEMO_RESPAWN_TIME = 3
SMALL_BOOST_RESPAWN_TIME = 4
BIG_BOOST_RESPAWN_TIME = 10


class PlayerData(Enum):
    IGNORE = 0  # Whether to mask out this player with attention
    MASK = 1  # Whether the data is masked out (e.g. everything else set to 0)
    TEAM = 2  # -1 for blue, +1 for orange
    POS_X = 3
    POS_Y = 4
    POS_Z = 5
    VEL_X = 6
    VEL_Y = 7
    VEL_Z = 8
    FW_X = 9
    FW_Y = 10
    FW_Z = 11
    UP_X = 12
    UP_Y = 13
    UP_Z = 14
    ANG_VEL_X = 15
    ANG_VEL_Y = 16
    ANG_VEL_Z = 17
    BOOST_AMOUNT = 18
    IS_DEMOED = 19
    RESPAWN_TIMER = 20
    # TODO consider adding jump/dodge/handbrake info


class BallData(Enum):
    IGNORE = 0
    MASK = 1
    POS_X = 2
    POS_Y = 3
    POS_Z = 4
    VEL_X = 5
    VEL_Y = 6
    VEL_Z = 7
    ANG_VEL_X = 8
    ANG_VEL_Y = 9
    ANG_VEL_Z = 10


class BoostData(Enum):
    IGNORE = 0
    MASK = 1
    TIMER_PAD_0 = 2
    TIMER_PAD_1 = 3
    TIMER_PAD_2 = 4
    TIMER_PAD_3 = 5
    TIMER_PAD_4 = 6
    TIMER_PAD_5 = 7
    TIMER_PAD_6 = 8
    TIMER_PAD_7 = 9
    TIMER_PAD_8 = 10
    TIMER_PAD_9 = 11
    TIMER_PAD_10 = 12
    TIMER_PAD_11 = 13
    TIMER_PAD_12 = 14
    TIMER_PAD_13 = 15
    TIMER_PAD_14 = 16
    TIMER_PAD_15 = 17
    TIMER_PAD_16 = 18
    TIMER_PAD_17 = 19
    TIMER_PAD_18 = 20
    TIMER_PAD_19 = 21
    TIMER_PAD_20 = 22
    TIMER_PAD_21 = 23
    TIMER_PAD_22 = 24
    TIMER_PAD_23 = 25
    TIMER_PAD_24 = 26
    TIMER_PAD_25 = 27
    TIMER_PAD_26 = 28
    TIMER_PAD_27 = 29
    TIMER_PAD_28 = 30
    TIMER_PAD_29 = 31
    TIMER_PAD_30 = 32
    TIMER_PAD_31 = 33
    TIMER_PAD_32 = 34
    TIMER_PAD_33 = 35
    TIMER_PAD_34 = 36

BIG_BOOST_INDEX = np.array([3, 4, 15, 18, 29, 30], dtype=np.int32)
MASK_BIG_BOOST = np.isin(np.arange(34), BIG_BOOST_INDEX)

def episode_to_data(states: [GameState], tick_skip) -> EpisodeData:
    time_per_tick = tick_skip / 120
    num_players = len(states[0].players)
    ep = EpisodeData.new_empty(len(states))
    demo_timers = [0] * num_players
    last_demoed = [False] * num_players
    for (i, player) in enumerate(states[0].players):
        demo_timers[i] = DEMO_RESPAWN_TIME if player.is_demoed else 0
        last_demoed[i] = True if player.is_demoed else False
    boost_timers = np.zeros(len(states[0].boost_pads))
    boost_timers[MASK_BIG_BOOST & (states[0].boost_pads == 0)] = BIG_BOOST_RESPAWN_TIME
    boost_timers[(~MASK_BIG_BOOST) & (states[0].boost_pads == 0)] = SMALL_BOOST_RESPAWN_TIME
    boost_timers[states[0].boost_pads == 1] = 0

    last_boost = np.array(states[0].boost_pads)

    for (i, state) in enumerate(states):
        # update demo timers and boost timers, skip first state because it was done above
        if i != 0:
            # demo timers
            for (k, player) in enumerate(states[0].players):
                if player.is_demoed and not last_demoed[k]:
                    demo_timers[k] = DEMO_RESPAWN_TIME
                elif demo_timers[k] > 0 and player.is_demoed:
                    demo_timers[k] -= time_per_tick
                elif not player.is_demoed:
                    demo_timers[k] = 0
                last_demoed[k] = True if player.is_demoed else False

            # boost timers
            # Subtract time_per_tick for unchanged boosts (from 0 to 0), make a minimum of 0 in case of timing issues
            boost_timers[(states[i].boost_pads == 0) & (last_boost == 0)] -= time_per_tick
            boost_timers[boost_timers < 0] = 0

            # Reset timers for active boosts
            boost_timers[states[i].boost_pads == 1] = 0

            # set changed from 1 to 0
            boost_timers[MASK_BIG_BOOST & (states[i].boost_pads == 0) & (last_boost == 1)] = BIG_BOOST_RESPAWN_TIME
            boost_timers[(~MASK_BIG_BOOST) & (states[i].boost_pads == 0) & (last_boost == 1)] = SMALL_BOOST_RESPAWN_TIME

            last_boost = np.array(states[i].boost_pads)

        ep.ball_data[i, 0, BallData.POS_X.value] = state.ball.position[0]
        ep.ball_data[i, 0, BallData.POS_Y.value] = state.ball.position[1]
        ep.ball_data[i, 0, BallData.POS_Z.value] = state.ball.position[2]
        ep.ball_data[i, 0, BallData.VEL_X.value] = state.ball.linear_velocity[0]
        ep.ball_data[i, 0, BallData.VEL_Y.value] = state.ball.linear_velocity[1]
        ep.ball_data[i, 0, BallData.VEL_Z.value] = state.ball.linear_velocity[2]
        ep.ball_data[i, 0, BallData.ANG_VEL_X.value] = state.ball.angular_velocity[0]
        ep.ball_data[i, 0, BallData.ANG_VEL_Y.value] = state.ball.angular_velocity[1]
        ep.ball_data[i, 0, BallData.ANG_VEL_Z.value] = state.ball.angular_velocity[2]

        for (j, player) in enumerate(state.players):
            ep.player_data[i, j, PlayerData.TEAM.value] = 2 * player.team_num - 1  # -1 for blue, 1 for orange
            ep.player_data[i, j, PlayerData.POS_X.value] = player.car_data.position[0]
            ep.player_data[i, j, PlayerData.POS_Y.value] = player.car_data.position[1]
            ep.player_data[i, j, PlayerData.POS_Z.value] = player.car_data.position[2]
            ep.player_data[i, j, PlayerData.VEL_X.value] = player.car_data.linear_velocity[0]
            ep.player_data[i, j, PlayerData.VEL_Y.value] = player.car_data.linear_velocity[1]
            ep.player_data[i, j, PlayerData.VEL_Z.value] = player.car_data.linear_velocity[2]
            quat = player.car_data.quaternion
            rot_mtx = quat_to_rot_mtx(quat)
            ep.player_data[i, j, PlayerData.FW_X.value] = rot_mtx[0, 0]
            ep.player_data[i, j, PlayerData.FW_Y.value] = rot_mtx[0, 1]
            ep.player_data[i, j, PlayerData.FW_Z.value] = rot_mtx[0, 2]
            ep.player_data[i, j, PlayerData.UP_X.value] = rot_mtx[2, 0]
            ep.player_data[i, j, PlayerData.UP_Y.value] = rot_mtx[2, 1]
            ep.player_data[i, j, PlayerData.UP_Z.value] = rot_mtx[2, 2]
            ep.player_data[i, j, PlayerData.ANG_VEL_X.value] = player.car_data.angular_velocity[0]
            ep.player_data[i, j, PlayerData.ANG_VEL_Y.value] = player.car_data.angular_velocity[1]
            ep.player_data[i, j, PlayerData.ANG_VEL_Z.value] = player.car_data.angular_velocity[2]
            ep.player_data[i, j, PlayerData.BOOST_AMOUNT.value] = player.boost_amount
            ep.player_data[i, j, PlayerData.IS_DEMOED.value] = player.is_demoed
            ep.player_data[i, j, PlayerData.RESPAWN_TIMER.value] = demo_timers[j]

        ep.boost_data[i, 0, 2:36] = boost_timers

    ep.normalize()

    return ep
