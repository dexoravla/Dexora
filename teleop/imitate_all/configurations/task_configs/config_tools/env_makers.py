def make_airbot_play3_env(config: dict):
    from envs.airbot_play_3_with_cam import AIRBOTPlayWithCameraEnv

    return AIRBOTPlayWithCameraEnv(config["env_config_path"])


def make_airbot_tok_2_env(config: dict):
    from envs.airbot_tok_2_env import AIRBOTTOKEnv

    return AIRBOTTOKEnv(config["env_config_path"])


def make_airbot_mmk2_env(config: dict):
    from envs.airbot_mmk_env import make_env

    return make_env(config)

def make_com_airbot_mmk2_env(config: dict):
    from envs.airbot_com_mmk_env import make_env

    return make_env(config)