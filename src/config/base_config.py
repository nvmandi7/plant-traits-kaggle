from argparse import ArgumentParser

from pydantic import BaseSettings, Field


class BaseConfig(BaseSettings):
    """
    BaseConfig is the base configuration class for training and inference.
    """

    config_file: str = Field(default=None, description="Path to configuration file.")

    @classmethod
    def parse_args(cls):
        parser = ArgumentParser()
        for field in cls.__fields__.values():
            parser.add_argument(
                f"--{field.name}",
                type=field.type_,
                default=field.default,
                help=field.field_info.description,
            )
        args = parser.parse_args()

        config = cls() if args.config_file is None else cls.parse_file(args.config_file)

        settings = config.dict()
        for key, value in args.__dict__.items():
            if value != config.__fields__[key].default:
                settings[key] = value

        return cls(**settings)

    class Config:
        """
        Config is used to configure how the BaseSettings class works.
        """

        # Allow any type to be used as a value in the configuration settings
        arbitrary_types_allowed = True

        # Allow arbitrary attributes to be added to the configuration settings
        extra = "allow"

        # Allow environment variables to be used to set configuration settings
        env_file = ".env"
