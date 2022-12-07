from gym.envs.registration import register

register(
    id="BoatNavi-v0",
    entry_point="boat_gym.envs.boat_navigation:BoatNaviEnv",
)