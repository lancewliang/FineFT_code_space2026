import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Transition_Dataset(Dataset):
    def __init__(self, states, infos, actions, rewards, next_states, next_infos, dones):
        self.states = states
        self.infos = infos
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.next_infos = next_infos
        self.dones = dones

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        selected_infos = {}
        for key in self.infos:
            selected_infos[key] = self.infos[key][idx]
        selected_infos_ = {}
        for key in self.next_infos:
            selected_infos_[key] = self.next_infos[key][idx]
        return (
            self.states[idx],
            selected_infos,
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            selected_infos_,
            self.dones[idx],
        )
if __name__ == "__main__":
    states = np.random.rand(100, 4)
    infos = {"info1": np.random.rand(100, 2), "info2": np.random.rand(100, 3)}
    actions = np.random.rand(100, 2)
    rewards = np.random.rand(100)
    next_states = np.random.rand(100, 4)
    next_infos = {"info1": np.random.rand(100, 2), "info2": np.random.rand(100, 3)}
    dones = np.random.randint(0, 2, 100)
    dataset = Transition_Dataset(states, infos, actions, rewards, next_states, next_infos, dones)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    for states, infos, actions, rewards, next_states, next_infos, dones in data_loader:
        print(states.shape, actions.shape, rewards.shape, dones.shape)
        for key in infos:
            print(key, infos[key].shape)
        for key in next_infos:
            print(key, next_infos[key].shape)
        break  
    