"""CSV ingestion for the News AI pipeline.

Exposes :class:`NewsParser`, which loads a news dataset from disk into a
pandas DataFrame for downstream chunking and embedding.
"""

import pandas as pd


class NewsParser:
    """Loads a news-article CSV file into a pandas DataFrame."""

    def __init__(self, csv_path: str):
        """Store the path to the source CSV.

        Args:
            csv_path: Path to the news dataset CSV file.
        """
        self.csv_path = csv_path

    def parse(self) -> pd.DataFrame:
        """Read the CSV file into a DataFrame.

        Returns:
            The parsed dataset as a pandas DataFrame.
        """
        df = pd.read_csv(self.csv_path)
        return df