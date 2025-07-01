from types import SimpleNamespace
import torch
from tensordict.tensordict import TensorDict
from tdmpc2.common.buffer import Buffer


def make_episode(length):
    # Create dummy episode with required fields
    obs = torch.zeros(length, 9, 64, 64)
    action = torch.zeros(length, 1)
    reward = torch.zeros(length)
    terminated = torch.zeros(length)
    td = TensorDict({
        "obs": obs,
        "action": action,
        "reward": reward,
        "terminated": terminated,
    }, batch_size=(length,))
    return td


def test_partial_eviction():
    # Create minimal config
    cfg = SimpleNamespace()
    cfg.buffer_size = 10
    cfg.steps = 10
    cfg.batch_size = 1
    cfg.horizon = 1
    cfg.q_sample_ratio = 0.5
    cfg.multitask = False
    cfg.save_obs_for_rankme = False

    buffer = Buffer(cfg)

    # Add episode of length 6 (E0)
    buffer.add(make_episode(6))
    # Add episode of length 3 (E1)
    buffer.add(make_episode(3))

    assert buffer._steps_in_buffer == 9
    assert len(buffer._episode_queue) == 2

    # Add episode of length 4 (E2) -> causes partial eviction (overwrite 3 steps)
    buffer.add(make_episode(4))

    # After addition, buffer full
    assert buffer._steps_in_buffer == buffer.capacity

    # The oldest episode should now have 3 steps remaining (6-3)
    oldest_id, oldest_rem, _ = buffer._episode_queue[0]
    assert oldest_rem == 3, f"Expected 3 remaining steps, got {oldest_rem}"
    assert len(buffer._episode_queue) == 3

    # Expected queue (oldest → newest):
    #   (E0 , 3 steps)
    #   (E1 , 3 steps)
    #   (E2 , 4 steps)

    # -------------- add episode of length 5 -----------------
    buffer.add(make_episode(5))  # E3

    # Buffer full again and queue length should still be 3 because we update
    # the head entry in-place when partially evicting.
    assert buffer._steps_in_buffer == buffer.capacity
    assert len(buffer._episode_queue) == 3

    # New expected queue:
    #   (E1 , 1 step)   – after partial eviction
    #   (E2 , 4 steps)
    #   (E3 , 5 steps)
    ids_rems = [(eid, rem) for eid, rem, _ in buffer._episode_queue]
    assert ids_rems[0][1] == 1, f"Oldest remaining steps wrong: {ids_rems}"
    assert ids_rems[1][1] == 4, f"Middle episode steps wrong: {ids_rems}"
    assert ids_rems[2][1] == 5, f"Newest episode steps wrong: {ids_rems}"


if __name__ == "__main__":
    test_partial_eviction()
    print("Partial eviction test passed.") 