from numpy import take
import create_griddly_env
import time
import env_util


def take_step(env, action, player_id):
    full_action = env_util.build_action(action, env.player_count, player_id)
    env.step(full_action)
    env.render(observer='global')
    print('Agent', player_id, ':', env_util.action_to_str(env, action))
    time.sleep(2)


if __name__=='__main__':
    env = create_griddly_env.create_griddly_env('collector_game.yaml')
    env.reset(level_string="w w w w w w w w w w w w w w w w\nw . s . . . . . . P1 . w . . . w\nw . . w . . P2 . s s . . . . . w\nw . . . . . . s . . . . . . . w\nw . . . s . . . . . . . . . . w\nw . w . . . w . . s . . . w . w\nw . . . s . . . . s . . . . . w\nw w . . . . . . . . . . . . . w\nw . . s . . s w . . . . s . . w\nw . . . . . . . . . . s . . s w\nw . . . . . . . . . . . . . w w\nw . . . . s . . . . . . . . . w\nw . . . . s . . . s . . w . . w\nw . . . . . . . . . . . . . . w\nw . w . . . . . . . . . . . . w\nw w w w w w w w w w w w w w w w")
    env.render(observer='global')
    time.sleep(1)
    take_step(env, [11, 1], 2)
    take_step(env, [0,1], 1)
    take_step(env, [0,1], 1)
    take_step(env, [0,1], 1)
    take_step(env, [6,1], 1)

    time.sleep(10)
