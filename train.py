from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.multiprocessing import Pipe

from agents import *
from config import *
from envs import *
from utils import *

import time

import csv


from gym_montezuma.envs import MontezumasRevengeEnv

def main():
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    assert train_method == 'RND'
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if 'skills' in env_id:
        env = MontezumasRevengeEnv()
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = False
    is_render = False
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    run_path = Path(f'runs/{env_id}_{datetime.now().strftime("%b%d_%H-%M-%S")}')
    log_path = run_path / 'logs'
    subgoals_path = run_path / 'subgoal_plots'
    run_path.mkdir(parents=True)
    log_path.mkdir()
    subgoals_path.mkdir()

    with open(run_path / 'step_data.csv','w+') as fd:
        #env_num, ep num, num option executions total, num actions executions total, ext op reward, done, real_done, action, player_pos, int_reward_per_one_decision
        csv_writer = csv.writer(fd, delimiter=',')
        csv_writer.writerow(['environment number','episode number', 'total option executions', 'total primitive action executions',
                'extrinsic option reward', 'done', 'real done', 'action', 'player position (x,y)', 'intrinsic reward for one decision'])

    with open(run_path / 'episode_data.csv','w+') as fd:
        #env_num, ep num, ep_rew, number of options, number of actions, int_reward_per_epi
        csv_writer = csv.writer(fd, delimiter=',')
        csv_writer.writerow(['environment number','episode number', 'episode reward', 'episode length options', 'episode length primitives', 'intrinsic reward per episode'])

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if use_cuda else 'torch.FloatTensor')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)

    agent = RNDAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )

    if is_load_model:
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
            agent.rnd.target.load_state_dict(torch.load(target_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
        print('load finished!')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    for _ in range(num_step * pre_obs_norm_step):
        for parent_conn in parent_conns:
            parent_conn.send('random')

        for parent_conn in parent_conns:
            s, r, d, rd, lr, _ = parent_conn.recv()
            next_obs.append(s[-1, :, :].reshape([1, 84, 84]))

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('End to initalize...')

    total_option_executions = 0
    total_primitive_executions = 0

    episode_counter = [0 for _ in range(num_worker)]
    episode_rewards = [0 for _ in range(num_worker)]
    episode_trajectories = [[] for _ in range(num_worker)]
    episode_length_primitives = [0 for _ in range(num_worker)]

    env_acting_times = [0 for _ in range(num_worker)]

    global_ep = 0

    while True:
        total_state, total_reward, total_done, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for cur_step in range(num_step):
            for parent_conn in parent_conns:
                parent_conn.send('get_available_actions')

            available_actions = []
            actions = []

            for i, parent_conn in enumerate(parent_conns):
                available_actions_list = parent_conn.recv()
                available_actions.append(available_actions_list)

            executed_actions = [None for _ in parent_conns]
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255., available_actions)

            for i, (parent_conn, action) in enumerate(zip(parent_conns, actions)):
                    episode_trajectories[i].append(action)
                    env_acting_times[i] = time.time()
                    parent_conn.send(action)
                    executed_actions[i] = action

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            for i,parent_conn in enuumerate(parent_conns):
                s, r, d, rd, lr, info = parent_conn.recv()
                env_acting_times[i] = time.time() - env_acting_times[i]

                episode_rewards[i] += r
                episode_length_primitives[i] += info['n_steps'] if 'n_steps' in info.keys() else 1
                total_option_executions += 1
                total_primitive_executions += info['n_steps'] if 'n_steps' in info.keys() else 1

                intrinsic_reward = agent.compute_intrinsic_reward(
                    ((s[-1, :, :].reshape([1, 84, 84]) - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))

                with open(run_path / 'step_data.csv','a+') as fd:
                    csv_writer = csv.writer(fd, delimiter=',')
                    if 'states' in info.keys():
                        csv_writer.writerow([i, episode_counter[i], total_option_executions, total_primitive_executions,
                                r, d, rd, executed_actions[i], str((info['states'][-1]['player_x'],info['states'][-1]['player_y'])), intrinsic_reward])
                    else:
                        csv_writer.writerow([i, episode_counter[i], total_option_executions, total_primitive_executions,
                                r, d, rd, executed_actions[i], "NA", intrinsic_reward])
                    fd.flush()

                if rd or d:
                    with open(run_path / 'episode_data.csv','a+') as fd:
                        csv_writer = csv.writer(fd, delimiter=',')
                        csv_writer.writerow([i, episode_counter[i], episode_rewards[i], len(episode_trajectories[i]), episode_length_primitives[i]])
                    episode_counter[i] += 1
                    episode_rewards[i] = 0
                    episode_trajectories[i] = []
                    global_ep += 1


                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs.append(s[-1, :, :].reshape([1, 84, 84]))

            print("Average number of actions per second: ", 1/np.mean(env_acting_times))

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)

            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0


        _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255., None)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_policy = np.vstack(total_policy_np)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        #total_int_reward = np.stack(total_int_reward).swapaxes(0, 1)
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              gamma,
                                              num_step,
                                              num_worker)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)

        # add ext adv and int adv
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy)

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(agent.rnd.target.state_dict(), target_path)


if __name__ == '__main__':
    main()
