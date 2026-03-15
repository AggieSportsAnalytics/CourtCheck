"""Performance optimization tests for the CourtCheck pipeline."""


def test_default_model_is_yolov8m():
    from backend.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    assert "yolov8m" in cfg.player_model, (
        f"Expected yolov8m model, got: {cfg.player_model}"
    )
