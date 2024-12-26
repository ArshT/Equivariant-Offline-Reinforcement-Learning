from bulletarm import env_factory

def runDemo():
  env_config = {'render': False, 'obs_type': 'vector'}
  env = env_factory.createEnvs(1, 'close_loop_drawer_opening', env_config)

  for i in range(10):
    obs = env.reset()
    print("Length of observation: ", len(obs))
    print("Observation Shape: ", obs[2].shape)
    done = False
    while not done:
      action = env.getNextAction()
      obs, reward, done = env.step(action)
  env.close()

if __name__ == '__main__':
  runDemo()
