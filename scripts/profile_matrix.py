from src.config import load_settings


if __name__ == "__main__":
    settings = load_settings()
    print(f"Active profile: {settings.profile}")
