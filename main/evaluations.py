#evaluations.py

def load_models(
    models_dir: str | Path = "models/",
    *,
    factory: Callable[[], nn.Module] | None = None,
) -> List[nn.Module]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models_path = Path(models_dir)
    checkpoints = sorted(models_path.glob(f"*{suffix}"))
    loaded: List[nn.Module] = []
    for ckpt in checkpoints:
     if factory is None:
        model = torch.load(ckpt, map_location=device)
     else:
        model = factory().to(device)
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(state_dict)

    model.name = ckpt.stem
    loaded.append(model)
    return loaded

def compare_models(
    models: List[nn.Module],
    test_loader,
    *,
    sort_by: str | Tuple[str, ...] = ("f1", "accuracy")
) -> pd.DataFrame:
    results: Dict[str, Dict[str, Any]] = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in models:
        model = model.to(device)
        summary, *_ = compute_metrics(model, test_loader, device=device)
        results[getattr(model, "name", model.__class__.__name__)] = {
            "average_loss": summary["average_loss"],
            "accuracy":     summary["accuracy"],
            "precision":    summary["precision"],
            "recall":       summary["recall"],
            "f1":           summary["f1"],
        }
    df = pd.DataFrame.from_dict(results, orient="index")

    if isinstance(sort_by, str):
        sort_by = (sort_by,)
    df = df.sort_values(list(sort_by), ascending=False)
    return df