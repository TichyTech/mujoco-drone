from gym.envs.registration import register

register(
    id="TestXML",
    max_episode_steps=1000,
    entry_point="env.TestXML:TestEnvXML",
)

register(
    id="Drones",
    max_episode_steps=1000,
    entry_point="env.Drones:DronesEnv",
)