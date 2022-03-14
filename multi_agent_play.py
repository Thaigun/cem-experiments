from random import choices
import pygame
from pygame.locals import VIDEORESIZE
import env_util
import configuration

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def play(env, agents_confs, cem, transpose=True, fps=30, zoom=None, keys_to_action=None, visualiser_callback=None):
    # Allows one to play the game using keyboard.
    rendered = env.render(observer="global", mode="rgb_array")

    if keys_to_action is not None:
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, (
                env.spec.id
                + " does not have explicit key to action mapping, "
                + "please specify one manually"
            )
        relevant_keys = set(sum(map(list, keys_to_action.keys()), []))
    else:
        relevant_keys = set()

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True
    info = {}

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    player_in_turn = 1
    trust_correction = False

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
            player_in_turn = 1
            continue
            
        current_policy = agents_confs[player_in_turn - 1].policy
        if current_policy != 'KBM':
            action_probs = current_policy(env, cem, player_in_turn)
            # Select one of the keys randomly, weighted by the values
            # I'm doing it like this because I'm scared the order won't be stable if I access the keys and values separately.
            action_probs_list = list(action_probs.items())
            keys = [x[0] for x in action_probs_list]
            probs = [x[1] for x in action_probs_list]
            action = choices(keys, weights=probs)[0]
            
            full_action = [[0,0] for _ in range(env.player_count)]
            full_action[player_in_turn-1] = list(action)
            action_desc = env_util.action_to_str(env, action)
            player_name = env_util.agent_id_to_name(agents_confs, player_in_turn)
            print(player_name, 'chose action', action_desc)
            obs, rew, env_done, info = env.step(full_action)
            player_in_turn = player_in_turn % env.player_count + 1
        else:
            # process pygame events
            for event in pygame.event.get():
                # test events, set key states
                if event.type == pygame.KEYDOWN:
                    if event.key in relevant_keys:
                        pressed_keys.append(event.key)
                    elif event.key == 27:
                        running = False
                    elif visualiser_callback is not None and event.key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('p')]:
                        c = chr(event.key)
                        if c == 'p':
                            visualiser_callback(env, agents_confs, trust_correction=trust_correction)
                        else:
                            visualiser_callback(env, agents_confs, int(c)-1, trust_correction=trust_correction)
                    elif event.key == ord('t'):
                        trust_correction = not trust_correction
                        print('Trust correction (for visualisations) is', trust_correction)
                    elif event.key == ord('y'):
                        configuration.set_health_performance_consistency(not configuration.health_performance_consistency)
                        print('Health performance consistency is', configuration.health_performance_consistency)
                elif event.type == pygame.KEYUP:
                    if event.key in relevant_keys:
                        pressed_keys.remove(event.key)
                elif event.type == pygame.QUIT:
                    running = False
                elif event.type == VIDEORESIZE:
                    video_size = event.size
                    screen = pygame.display.set_mode(video_size)
                    print(video_size)

            action = keys_to_action.get(tuple(sorted(pressed_keys)), None)
            if action != None:
                full_action = [[0,0] for _ in range(env.player_count)]
                full_action[player_in_turn - 1] = action
                obs, rew, env_done, info = env.step(full_action)
                player_in_turn = player_in_turn % env.player_count + 1        

        if obs is not None:
            rendered = env.render(observer="global", mode="rgb_array")
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
