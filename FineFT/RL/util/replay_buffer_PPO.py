import torch
import numpy as np
import copy


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(
            self.a, dtype=torch.long
        )  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done


# the one we are using, three sequence input and 1 single input state_seq, previous_price_change_seq,preivous_startegy_reward_seq, previous_action
class ReplayBuffer_high_level:
    def __init__(
        self, batch_size, seq_length, state_dim, reward_dim, pre_act_dim, device="cpu"
    ):
        self.state_seq = np.zeros((batch_size, seq_length, state_dim))
        self.reward_seq = np.zeros((batch_size, seq_length, reward_dim))
        self.price_change_seq = np.zeros((batch_size, seq_length, reward_dim))
        self.previous_action = np.zeros((batch_size, pre_act_dim))

        self.a_index = np.zeros((batch_size, 1))
        self.a_logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        self.state_seq_ = np.zeros((batch_size, seq_length, state_dim))
        self.reward_seq_ = np.zeros((batch_size, seq_length, reward_dim))
        self.price_change_seq_ = np.zeros((batch_size, seq_length, reward_dim))
        self.previous_action_ = np.zeros((batch_size, pre_act_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0
        self.device = device

    def store(
        self,
        state_seq,
        reward_seq,
        price_change_seq,
        previous_action,
        a_index,
        a_logprob,
        r,
        state_seq_,
        reward_seq_,
        price_change_seq_,
        previous_action_,
        dw,
        done,
    ):
        self.state_seq[self.count] = state_seq
        self.reward_seq[self.count] = reward_seq
        self.price_change_seq[self.count] = price_change_seq
        self.previous_action[self.count] = previous_action

        self.a_index[self.count] = a_index
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r

        self.state_seq_[self.count] = state_seq_
        self.reward_seq_[self.count] = reward_seq_
        self.price_change_seq_[self.count] = price_change_seq_
        self.previous_action_[self.count] = previous_action_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        state_seq = torch.tensor(self.state_seq, dtype=torch.float).to(self.device)
        reward_seq = torch.tensor(self.reward_seq, dtype=torch.float).to(self.device)
        price_change_seq = torch.tensor(self.price_change_seq, dtype=torch.float).to(
            self.device
        )
        previous_action = torch.tensor(self.previous_action, dtype=torch.float).to(
            self.device
        )

        a_index = torch.tensor(self.a_index, dtype=torch.long).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)

        state_seq_ = torch.tensor(self.state_seq_, dtype=torch.float).to(self.device)
        reward_seq_ = torch.tensor(self.reward_seq_, dtype=torch.float).to(self.device)
        price_change_seq_ = torch.tensor(self.price_change_seq_, dtype=torch.float).to(
            self.device
        )
        previous_action_ = torch.tensor(self.previous_action_, dtype=torch.float).to(
            self.device
        )

        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)

        return (
            state_seq,
            reward_seq,
            price_change_seq,
            previous_action,
            a_index,
            a_logprob,
            r,
            state_seq_,
            reward_seq_,
            price_change_seq_,
            previous_action_,
            dw,
            done,
        )


# for reward only sequence
class ReplayBuffer_high_level_reward_seq:
    def __init__(
        self,
        batch_size,
        seq_length,
        reward_dim,
        pre_act_dim,
        action_dim=1,
        device="cpu",
    ):
        self.reward_seq = np.zeros((batch_size, seq_length, reward_dim))
        self.price_change_seq = np.zeros((batch_size, seq_length, reward_dim))
        self.previous_action = np.zeros((batch_size, pre_act_dim))

        self.a_index = np.zeros((batch_size, action_dim))
        self.a_logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))

        self.reward_seq_ = np.zeros((batch_size, seq_length, reward_dim))
        self.price_change_seq_ = np.zeros((batch_size, seq_length, reward_dim))
        self.previous_action_ = np.zeros((batch_size, pre_act_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0
        self.device = device

    def store(
        self,
        reward_seq,
        price_change_seq,
        previous_action,
        a_index,
        a_logprob,
        r,
        reward_seq_,
        price_change_seq_,
        previous_action_,
        dw,
        done,
    ):
        self.reward_seq[self.count] = reward_seq
        self.price_change_seq[self.count] = price_change_seq
        self.previous_action[self.count] = previous_action

        self.a_index[self.count] = a_index
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r

        self.reward_seq_[self.count] = reward_seq_
        self.price_change_seq_[self.count] = price_change_seq_
        self.previous_action_[self.count] = previous_action_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        reward_seq = torch.tensor(self.reward_seq, dtype=torch.float).to(self.device)
        price_change_seq = torch.tensor(self.price_change_seq, dtype=torch.float).to(
            self.device
        )
        previous_action = torch.tensor(self.previous_action, dtype=torch.float).to(
            self.device
        )

        a_index = torch.tensor(self.a_index, dtype=torch.long).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)

        reward_seq_ = torch.tensor(self.reward_seq_, dtype=torch.float).to(self.device)
        price_change_seq_ = torch.tensor(self.price_change_seq_, dtype=torch.float).to(
            self.device
        )
        previous_action_ = torch.tensor(self.previous_action_, dtype=torch.float).to(
            self.device
        )

        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)

        return (
            reward_seq,
            price_change_seq,
            previous_action,
            a_index,
            a_logprob,
            r,
            reward_seq_,
            price_change_seq_,
            previous_action_,
            dw,
            done,
        )


class ReplayBuffer_Multi_info:
    def __init__(self, batch_size, state_dim, action_dim, device="cuda"):
        self.s = np.zeros((batch_size, state_dim))
        self.previous_action = np.zeros((batch_size, 1))
        self.avaliable_action = np.zeros((batch_size, action_dim))
        self.q_value = np.zeros((batch_size, action_dim))

        self.a = np.zeros((batch_size, 1))
        self.a_logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        self.all_act_prob = np.zeros((batch_size, action_dim))

        self.s_ = np.zeros((batch_size, state_dim))
        self.previous_action_ = np.zeros((batch_size, 1))
        self.avaliable_action_ = np.zeros((batch_size, action_dim))
        self.q_value_ = np.zeros((batch_size, action_dim))

        self.done = np.zeros((batch_size, 1))
        self.count = 0
        self.device = device

    def store(
        self,
        s,
        previous_action,
        avaliable_action,
        q_value,
        a,
        a_logprob,
        r,
        all_act_prob,
        s_,
        previous_action_,
        avaliable_action_,
        q_value_,
        done,
    ):
        self.s[self.count] = s
        self.previous_action[self.count] = previous_action
        self.avaliable_action[self.count] = avaliable_action
        self.q_value[self.count] = q_value

        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.all_act_prob[self.count] = all_act_prob

        self.s_[self.count] = s_
        self.previous_action_[self.count] = previous_action_
        self.avaliable_action_[self.count] = avaliable_action_
        self.q_value_[self.count] = q_value_
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        previous_action = torch.tensor(self.previous_action, dtype=torch.float).to(
            self.device
        )
        avaliable_action = torch.tensor(self.avaliable_action, dtype=torch.float).to(
            self.device
        )
        q_value = torch.tensor(self.q_value, dtype=torch.float).to(self.device)

        a = torch.tensor(self.a, dtype=torch.long).to(
            self.device
        )  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        all_act_prob = torch.tensor(self.all_act_prob, dtype=torch.float).to(
            self.device
        )

        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        previous_action_ = torch.tensor(self.previous_action_, dtype=torch.float).to(
            self.device
        )
        avaliable_action_ = torch.tensor(self.avaliable_action_, dtype=torch.float).to(
            self.device
        )
        q_value_ = torch.tensor(self.q_value_, dtype=torch.float).to(self.device)

        done = torch.tensor(self.done, dtype=torch.float).to(self.device)

        return (
            s,
            previous_action,
            avaliable_action,
            q_value,
            a,
            a_logprob,
            r,
            all_act_prob,
            s_,
            previous_action_,
            avaliable_action_,
            q_value_,
            done,
        )


class ReplayBuffer_lstm:
    def __init__(self, gamma, lamda, state_dim, action_dim, episode_limit, batch_size):
        self.gamma = gamma
        self.lamda = lamda
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_limit = episode_limit
        self.batch_size = batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            "s": np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
            "p_a": np.zeros([self.batch_size, self.episode_limit, 1]),
            "a_a": np.zeros([self.batch_size, self.episode_limit, self.action_dim]),
            "v": np.zeros([self.batch_size, self.episode_limit + 1]),
            "a": np.zeros([self.batch_size, self.episode_limit]),
            "a_logprob": np.zeros([self.batch_size, self.episode_limit]),
            "r": np.zeros([self.batch_size, self.episode_limit]),
            "dw": np.ones(
                [self.batch_size, self.episode_limit]
            ),  # Note: We use 'np.ones' to initialize 'dw'
            "active": np.zeros([self.batch_size, self.episode_limit]),
        }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, s, p_a, a_a, v, a, a_logprob, r, dw):

        self.buffer["s"][self.episode_num][episode_step] = s
        self.buffer["p_a"][self.episode_num][episode_step] = p_a
        self.buffer["a_a"][self.episode_num][episode_step] = a_a

        self.buffer["v"][self.episode_num][episode_step] = v
        self.buffer["a"][self.episode_num][episode_step] = a
        self.buffer["a_logprob"][self.episode_num][episode_step] = a_logprob
        self.buffer["r"][self.episode_num][episode_step] = r
        self.buffer["dw"][self.episode_num][episode_step] = dw

        self.buffer["active"][self.episode_num][episode_step] = 1.0

    def store_last_value(self, episode_step, v):
        self.buffer["v"][self.episode_num][episode_step] = v
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_adv(self):
        # Calculate the advantage using GAE
        v = self.buffer["v"][:, : self.max_episode_len]
        v_next = self.buffer["v"][:, 1 : self.max_episode_len + 1]
        r = self.buffer["r"][:, : self.max_episode_len]
        dw = self.buffer["dw"][:, : self.max_episode_len]
        active = self.buffer["active"][:, : self.max_episode_len]
        adv = np.zeros_like(r)  # adv.shape=(batch_size,max_episode_len)
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len)
            deltas = r + self.gamma * v_next * (1 - dw) - v
            for t in reversed(range(self.max_episode_len)):
                gae = (
                    deltas[:, t] + self.gamma * self.lamda * gae
                )  # gae.shape=(batch_size)
                adv[:, t] = gae
            v_target = adv + v  # v_target.shape(batch_size,max_episode_len)
            adv_copy = copy.deepcopy(adv)
            adv_copy[active == 0] = np.nan  # 忽略掉active=0的那些adv
            # adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
        return adv, v_target

    def get_training_data(self):
        adv, v_target = self.get_adv()
        batch = {
            "s": torch.tensor(
                self.buffer["s"][:, : self.max_episode_len], dtype=torch.float32
            ),
            "p_a": torch.tensor(
                self.buffer["p_a"][:, : self.max_episode_len], dtype=torch.float32
            ),
            "a_a": torch.tensor(
                self.buffer["a_a"][:, : self.max_episode_len], dtype=torch.float32
            ),
            "a": torch.tensor(
                self.buffer["a"][:, : self.max_episode_len], dtype=torch.long
            ),  # 动作a的类型必须是long
            "a_logprob": torch.tensor(
                self.buffer["a_logprob"][:, : self.max_episode_len], dtype=torch.float32
            ),
            "active": torch.tensor(
                self.buffer["active"][:, : self.max_episode_len], dtype=torch.float32
            ),
            "adv": torch.tensor(adv, dtype=torch.float32),
            "v_target": torch.tensor(v_target, dtype=torch.float32),
        }

        return batch
