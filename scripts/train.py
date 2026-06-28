from __future__ import annotations

import argparse

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.trainers import Trainer, TrainingConfig
from src.utils import seed_everything


def _parse_float_list(value: str) -> list[float]:
    values = [part.strip() for part in value.split(",") if part.strip() != ""]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated float values")
    try:
        return [float(part) for part in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated float values") from error


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a recommendation model on a manifest dataset.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparse-lr", type=float, default=None)
    parser.add_argument("--task-weights", type=_parse_float_list, default=None)
    parser.add_argument("--scenario-weights", type=_parse_float_list, default=None)
    parser.add_argument("--disable-data-validation", dest="validate_data", action="store_false")
    parser.add_argument("--validation-max-rows", type=int, default=1000)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--token-dim", type=int, default=36)
    parser.add_argument("--feature-backbone", choices=["rankmixer", "attention"], default="rankmixer")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ffn-type", choices=["dense", "sparse_moe"], default="dense")
    parser.add_argument("--sparse-moe-num-experts", type=int, default=4)
    parser.add_argument("--sparse-moe-loss-weight", type=float, default=0.0)
    parser.add_argument("--sparse-moe-target-active-ratio", type=float, default=None)
    parser.add_argument("--sparse-moe-loss-weight-update-rate", type=float, default=0.05)
    parser.add_argument("--sparse-moe-loss-weight-min", type=float, default=0.0)
    parser.add_argument("--sparse-moe-loss-weight-max", type=float, default=1.0)
    parser.add_argument("--sparse-moe-dtsi-infer-weight", type=float, default=0.5)
    parser.add_argument("--sparse-moe-inference-threshold", type=float, default=0.0)
    parser.add_argument("--disable-sparse-moe-dtsi", dest="sparse_moe_use_dtsi", action="store_false")
    parser.add_argument("--disable-task-tokens", dest="use_task_tokens", action="store_false")
    parser.add_argument("--disable-scenario-tokens", dest="use_scenario_tokens", action="store_false")
    parser.add_argument("--disable-global-scenario-token", dest="use_global_scenario_token", action="store_false")
    parser.add_argument(
        "--disable-task-feature-interaction",
        dest="use_task_feature_interaction",
        action="store_false",
    )
    parser.add_argument(
        "--disable-scenario-feature-interaction",
        dest="use_scenario_feature_interaction",
        action="store_false",
    )
    parser.set_defaults(
        sparse_moe_use_dtsi=True,
        use_task_tokens=True,
        use_scenario_tokens=True,
        use_global_scenario_token=True,
        use_task_feature_interaction=True,
        use_scenario_feature_interaction=True,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-path", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    trainer = Trainer(
        TrainingConfig(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            eval_max_batches=args.eval_max_batches,
            device=args.device,
            lr=args.lr,
            sparse_lr=args.sparse_lr,
            task_weights=args.task_weights,
            scenario_weights=args.scenario_weights,
            validate_data=args.validate_data,
            validation_max_rows=args.validation_max_rows,
            embedding_dim=args.embedding_dim,
            token_dim=args.token_dim,
            feature_backbone=args.feature_backbone,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_hidden_dim=args.ffn_hidden_dim,
            dropout=args.dropout,
            ffn_type=args.ffn_type,
            sparse_moe_num_experts=args.sparse_moe_num_experts,
            sparse_moe_loss_weight=args.sparse_moe_loss_weight,
            sparse_moe_target_active_ratio=args.sparse_moe_target_active_ratio,
            sparse_moe_loss_weight_update_rate=args.sparse_moe_loss_weight_update_rate,
            sparse_moe_loss_weight_min=args.sparse_moe_loss_weight_min,
            sparse_moe_loss_weight_max=args.sparse_moe_loss_weight_max,
            sparse_moe_use_dtsi=args.sparse_moe_use_dtsi,
            sparse_moe_dtsi_infer_weight=args.sparse_moe_dtsi_infer_weight,
            sparse_moe_inference_threshold=args.sparse_moe_inference_threshold,
            use_task_tokens=args.use_task_tokens,
            use_scenario_tokens=args.use_scenario_tokens,
            use_global_scenario_token=args.use_global_scenario_token,
            use_task_feature_interaction=args.use_task_feature_interaction,
            use_scenario_feature_interaction=args.use_scenario_feature_interaction,
            checkpoint_path=args.checkpoint_path,
        )
    )
    trainer.train()


if __name__ == "__main__":
    main()
