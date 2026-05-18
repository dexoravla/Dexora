from dataclasses import dataclass, field, replace
from robots.airbots.airbot_play.airbot_play_2_demonstration import (
    AIRBOTPlayDemonstrationConfig,
    AIRBOTPlayDemonstration,
)
from robots.airbots.airbot_base.airbot_base import AIRBOTBase, AIRBOTBaseConfig
from typing import Optional


@dataclass
class AIRBOTTOKDemonstrationConfig(object):
    airbot_play_demonstration: AIRBOTPlayDemonstrationConfig = field(
        default_factory=AIRBOTPlayDemonstrationConfig
    )
    base: AIRBOTBaseConfig = field(default_factory=AIRBOTBaseConfig)


class AIRBOTTOKDemonstration(object):
    def __init__(
        self, config: Optional[AIRBOTTOKDemonstrationConfig] = None, **kwargs
    ) -> None:
        if config is None:
            config = AIRBOTTOKDemonstrationConfig()
        self.config = replace(config, **kwargs)
        if self.config.base is not None:
            self.airbot_base = AIRBOTBase(self.config.base)
        self.airbot_play_demon = AIRBOTPlayDemonstration(
            self.config.airbot_play_demonstration
        )
        self.logs = {}
        self.cameras = self.airbot_play_demon.cameras
        print(self.config)

    def reset(self):
        self.airbot_play_demon.reset()

    def enter_active_mode(self):
        self.airbot_play_demon.enter_active_mode()

    def enter_passive_mode(self):
        self.airbot_play_demon.enter_passive_mode()

    def get_low_dim_data(self):
        data = self.airbot_play_demon.get_low_dim_data()
        data["action/base/velocity"] = self.airbot_base.get_current_velocity2D()
        return data

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        obs_act_dict = self.airbot_play_demon.capture_observation()

        if self.config.base is not None:
            obs_act_dict["low_dim"]["observation/base/velocity"] = list(
                self.airbot_base.get_current_velocity2D()
            )

        return obs_act_dict

    def exit(self):
        return self.airbot_play_demon.exit()

    def get_state_mode(self):
        return self.airbot_play_demon.get_state_mode()
