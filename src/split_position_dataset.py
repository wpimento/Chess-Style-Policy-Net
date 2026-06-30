from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/position_dataset.parquet")

TRAIN_PATH = Path("data/processed/train_positions.parquet")
VALIDATION_PATH = Path("data/processed/validation_positions.parquet")
TEST_PATH = Path("data/processed/test_positions.parquet")

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

def split_position_dataset() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Position dataset does not exist: {INPUT_PATH.resolve()}"
        )
    
    positions = pd.read_parquet(INPUT_PATH)

    required_columns = {
        "game_uuid",
        "end_time"
    }

    missing_columns = required_columns - set(positions.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    
    if positions["game_uuid"].isna().any():
        raise ValueError("Some positions have missing game UUIDs")

    if positions["end_time"].isna().any():
        raise ValueError("Some positions have missing end times")
    
    # chronological ordering; one row per game
    games = (
        positions[["game_uuid", "end_time"]]
        .drop_duplicates(subset="game_uuid")
        .sort_values(
            by=["end_time", "game_uuid"],
            kind="stable"
        )
        .reset_index(drop=True)
    )

    num_games = len(games)

    # find endpoints for each split
    #? helps with indexing
    train_end = int(num_games * TRAIN_RATIO)
    validation_end = train_end + int(
        num_games * VALIDATION_RATIO
    )

    # get uuids for cross referencing
    train_game_ids = set(
        games.iloc[:train_end]["game_uuid"]
    )

    validation_game_ids = set(
        games.iloc[train_end:validation_end]["game_uuid"]
    )

    test_game_ids = set(
        games.iloc[validation_end:]["game_uuid"]
    )

    train_positions = positions[
        positions["game_uuid"].isin(train_game_ids)
    ].copy()

    validation_positions = positions[
        positions["game_uuid"].isin(validation_game_ids)
    ].copy()

    test_positions = positions[
        positions["game_uuid"].isin(test_game_ids)
    ].copy()

    #! check for leaks
    assert train_game_ids.isdisjoint(validation_game_ids)
    assert train_game_ids.isdisjoint(test_game_ids)
    assert validation_game_ids.isdisjoint(test_game_ids)

    assert (
        len(train_positions)
        + len(validation_positions)
        + len(test_positions)
        == len(positions)
    )

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_positions.to_parquet(TRAIN_PATH, index=False)
    validation_positions.to_parquet(
        VALIDATION_PATH,
        index=False,
    )
    test_positions.to_parquet(TEST_PATH, index=False)

    print_split_summary(
        name="Train",
        positions=train_positions,
    )

    print_split_summary(
        name="Validation",
        positions=validation_positions,
    )

    print_split_summary(
        name="Test",
        positions=test_positions,
    )

    print(f"\nSaved: {TRAIN_PATH}")
    print(f"Saved: {VALIDATION_PATH}")
    print(f"Saved: {TEST_PATH}")

# helper for displaying results
def print_split_summary(
    name: str,
    positions: pd.DataFrame,
) -> None:
    number_of_positions = len(positions)
    number_of_games = positions["game_uuid"].nunique()

    earliest_time = positions["end_time"].min()
    latest_time = positions["end_time"].max()

    print(f"\n{name}:")
    print(f"  Games: {number_of_games:,}")
    print(f"  Positions: {number_of_positions:,}")
    print(f"  Earliest end_time: {earliest_time}")
    print(f"  Latest end_time: {latest_time}")

if __name__ == "__main__":
    split_position_dataset()