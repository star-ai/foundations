import unittest
import gym


class TestGymEnvs(unittest.TestCase):


  def run_env(self, env_name):
    env = gym.make(env_name)
    observation = env.reset()
    for t in range(100):
      env.render()
      # print(observation)
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      print("obs:", observation, "reward:", reward)
      if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

  def test_pendulum(self):
    self.run_env("Pendulum-v0")


if __name__ == '__main__':
    unittest.main()