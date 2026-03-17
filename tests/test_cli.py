"""Tests for the PrimateFace CLI."""

from primateface.cli import main


class TestCLIModels:
    """Test 'primateface models' command."""

    def test_models_list(self, capsys):
        ret = main(["models", "list"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "fparodi/primateface-models" in captured.out
        assert "detection" in captured.out
        assert "pose" in captured.out
        assert "hrnet" in captured.out.lower() or "HRNet" in captured.out

    def test_models_default_action(self, capsys):
        ret = main(["models"])
        assert ret == 0


class TestCLIHelp:
    """Test CLI help and no-args behavior."""

    def test_no_args_returns_zero(self, capsys):
        ret = main([])
        assert ret == 0

    def test_analyze_missing_input(self):
        """analyze without input should fail."""
        try:
            main(["analyze"])
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code == 2  # argparse error
