import os

def get_engine(engine_type: str = None, output_dir: str = "outputs"):
    engine_type = engine_type or os.getenv("TTS_ENGINE", "gtts")

    if engine_type == "pyttsx3":
        from tts.engines.pyttsx3_engine import Pyttsx3Engine
        return Pyttsx3Engine(output_dir=output_dir)
    elif engine_type == "gtts":
        from tts.engines.gtts_engine import GTTSEngine
        return GTTSEngine(output_dir=output_dir)
    elif engine_type == "google_cloud":
        from tts.engines.google_cloud_engine import GoogleCloudTTSEngine
        return GoogleCloudTTSEngine(output_dir=output_dir)
    elif engine_type == "elevenlabs":
        from tts.engines.elevenlabs_engine import ElevenLabsEngine
        return ElevenLabsEngine(output_dir=output_dir)
    else:
        raise ValueError(f"Unknown TTS engine: {engine_type!r}")
