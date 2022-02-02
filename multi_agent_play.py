from random import choices
import pygame
from pygame.locals import VIDEORESIZE

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def play(env, agent_policies, cem, transpose=True, fps=30, zoom=None, keys_to_action=None, visualiser_callback=None):
    """Allows one to play the game using keyboard.

    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            env: the environment
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    rendered = env.render(observer="global", mode="rgb_array")

    if keys_to_action is None:
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

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
            player_in_turn = 1
            continue
            
        current_policy = agent_policies[player_in_turn]
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
            print('Agent ', str(player_in_turn), ' chose action ', action)
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
                    elif visualiser_callback is not None and event.key == ord('p'):
                        visualiser_callback(env)
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
