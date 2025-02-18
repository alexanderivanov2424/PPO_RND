import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, RNDModel
from utils import global_grad_norm_


class RNDAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.rnd = RNDModel(input_size, output_size)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()),
                                    lr=learning_rate)
        self.rnd = self.rnd.to(self.device)

        self.model = self.model.to(self.device)

    def get_action(self, state, available_actions):
        available_actions = np.array(available_actions, dtype=float)
        state = torch.tensor(state, dtype=torch.float)
        policy, value_ext, value_int = self.model(state)
        if available_actions is not None and np.sum(policy.detach().numpy()) == 0:
            action = np.array([np.random.choice(np.where(actions==1)[0]) for actions in available_actions])
        elif available_actions is not None:
            action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()
            action_prob = action_prob * available_actions
            action_prob /= np.sum(action_prob, axis=1)[:,None]
            action = self.random_choice_prob_index(action_prob)
        else:
            action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()
            action = self.random_choice_prob_index(action_prob)
        return action, value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.tensor(next_obs, dtype=torch.float)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def train_only_rnd(self, s_batch, next_obs_batch):
        s_batch = torch.tensor(s_batch, dtype=torch.float)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')
        for _ in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)) < self.update_proportion
                loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1]))
                # ---------------------------------------------------------------------------------

                self.optimizer.zero_grad()
                loss.backward()
                global_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor.parameters()))
                self.optimizer.step()

    def train_only_ppo(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        s_batch = torch.tensor(s_batch, dtype=torch.float)
        target_ext_batch = torch.tensor(target_ext_batch, dtype=torch.float)
        target_int_batch = torch.tensor(target_int_batch, dtype=torch.float)
        y_batch = torch.tensor(y_batch, dtype=torch.long)
        adv_batch = torch.tensor(adv_batch, dtype=torch.float)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for _ in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                #critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy
                loss.backward()
                global_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor.parameters()))
                self.optimizer.step()


    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        s_batch = torch.tensor(s_batch, dtype=torch.float)
        target_ext_batch = torch.tensor(target_ext_batch, dtype=torch.float)
        target_int_batch = torch.tensor(target_int_batch, dtype=torch.float)
        y_batch = torch.tensor(y_batch, dtype=torch.long)
        adv_batch = torch.tensor(adv_batch, dtype=torch.float)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for _ in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)) < self.update_proportion
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1]))
                # ---------------------------------------------------------------------------------

                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss
                loss.backward()
                global_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor.parameters()))
                self.optimizer.step()
