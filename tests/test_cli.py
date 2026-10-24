from carmusic.cli import main


def test_situations_lists_every_dimension(capsys):
    assert main(["situations"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 9
    assert lines[-1].startswith("(or 'none'")


def test_recommend_prints_k_songs(capsys):
    code = main(["recommend", "--user", "1001", "--situation", "traffic jam", "-k", "3", "--model", "biases"])
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert [line.split(".")[0].strip() for line in out] == ["1", "2", "3"]


def test_recommend_rejects_unknown_input(capsys):
    assert main(["recommend", "--user", "1", "--situation", "traffic jam"]) == 2
    assert main(["recommend", "--user", "1001", "--situation", "on the moon"]) == 2


def test_evaluate_prints_a_markdown_table(capsys):
    args = ["evaluate", "--protocol", "new-song", "--models", "global-mean,seen-before", "--folds", "2"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "| global-mean |" in out and "| seen-before |" in out
    assert "| model | MAE | RMSE | nDCG@10 | pair acc. |" in out
