from griddly.RenderTools import VideoRecorder
from create_griddly_env import create_griddly_env
import os


def make_video_from_data(game_run_obj, sub_dir, name, frames_per_state=1):
    video_dir = os.path.join('video_exports', sub_dir)
    ensure_directory_exists(video_dir)
    video_recorder = VideoRecorder()
    env = create_griddly_env(game_run_obj['GriddlyDescription'])
    env.reset(level_string=game_run_obj['Map'])
    global_visu = env.render(observer='global', mode='rgb_array')
    video_recorder.start(os.path.join(video_dir, name + '.mp4'), global_visu.shape)

    for action in game_run_obj['Actions']:
        obs, reward, done, info = env.step(action)
        frame = env.render(observer='global', mode='rgb_array')
        for _ in range(frames_per_state):
            video_recorder.add_frame(frame)

    video_recorder.close()


def ensure_directory_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
