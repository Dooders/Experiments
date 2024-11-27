"""
ExperimentTracker: A comprehensive tool for managing and analyzing simulation experiments.

This module provides functionality to track, compare, and visualize multiple simulation runs.
It stores experiment metadata in JSON format and experimental results in SQLite databases,
allowing for efficient storage and retrieval of experiment configurations and metrics.

Key features:
- Experiment registration and metadata management
- Comparative analysis of multiple experiments
- Automated report generation with visualizations
- Statistical summaries of experiment results
- Data export capabilities
- Automatic cleanup of old experiments

Typical usage:
    tracker = ExperimentTracker("experiments")
    exp_id = tracker.register_experiment("test_run", config_dict, "data.db")
    tracker.generate_comparison_report([exp_id1, exp_id2])
"""

import csv
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader

# Add logging configuration at the module level
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ExperimentTracker:
    """
    A class for tracking and analyzing simulation experiments.

    This class manages experiment metadata, facilitates comparison between different
    experiment runs, and generates detailed reports with visualizations.

    Attributes
    ----------
    experiments_dir (Path):
        Directory where experiment data is stored
    metadata_file (Path):
        Path to the JSON file storing experiment metadata
    metadata (dict):
        Dictionary containing all experiment metadata
    template_env (Environment):
        Jinja2 environment for template rendering

    Methods
    -------
    register_experiment(self, name: str, config: Dict[str, Any], db_path: Path | str) -> str:
        Register a new experiment run with its configuration and database location.
    compare_experiments(self, experiment_ids: List[str], metrics: Optional[List[str]] = None, fill_method: str = 'nan') -> pd.DataFrame:
        Compare metrics across multiple experiments with graceful handling of missing data.
    generate_comparison_report(self, experiment_ids: List[str], output_file: Path | str | None = None):
        Generate a comprehensive HTML report comparing multiple experiments.
    generate_comparison_summary(self, experiment_ids: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        Generate a summary of the comparison including missing data statistics.
    export_experiment_data(self, experiment_id: str, output_path: Path | str):
        Export experiment data to CSV format.
    cleanup_old_experiments(self, days_old: int = 30):
        Remove experiments older than specified days.
    """

    def __init__(self, experiments_dir: Path | str = "experiments") -> None:
        """
        Initialize the ExperimentTracker.

        Parameters
        ----------
            experiments_dir (Path | str): Path to the directory where experiment data will be stored
        """
        self.experiments_dir = Path(experiments_dir)
        try:
            self.experiments_dir.mkdir(exist_ok=True)
        except PermissionError as e:
            logging.error(f"Failed to create experiments directory: {e}")
            raise
        self.metadata_file = self.experiments_dir / "metadata.json"
        self._load_metadata()

        # Initialize Jinja environment
        self.template_env = Environment(
            loader=FileSystemLoader("templates"), autoescape=True
        )

    def _load_metadata(self) -> None:
        """Load or create experiment metadata."""
        try:
            if self.metadata_file.exists():
                with self.metadata_file.open("r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    logging.debug("Metadata loaded successfully")
            else:
                self.metadata = {"experiments": {}}
                self._save_metadata()
                logging.info("Created new metadata file")
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to load metadata: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading metadata: {e}")
            raise

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        try:
            # Create a backup of the existing metadata file
            if self.metadata_file.exists():
                backup_path = self.metadata_file.with_suffix(".json.bak")
                self.metadata_file.rename(backup_path)
                logging.debug("Created metadata backup")

            # Write new metadata
            with self.metadata_file.open("w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
                logging.debug("Metadata saved successfully")

            # Remove backup if save was successful
            if backup_path.exists():
                backup_path.unlink()
        except IOError as e:
            logging.error(f"Failed to save metadata: {e}")
            # Restore from backup if available
            if "backup_path" in locals() and backup_path.exists():
                backup_path.rename(self.metadata_file)
                logging.info("Restored metadata from backup")
            raise
        except Exception as e:
            logging.error(f"Unexpected error saving metadata: {e}")
            raise

    def register_experiment(
        self, name: str, config: Dict[str, Any], db_path: Path | str
    ) -> str:
        """
        Register a new experiment run with its configuration and database location.

        Parameters
        ----------
        name : str
            Human-readable name for the experiment
        config : Dict[str, Any]
            Configuration parameters used for the experiment
        db_path : Path | str
            Path to the SQLite database containing experiment results

        Returns
        -------
        str
            Unique experiment identifier (UUID)

        Raises
        ------
        ValueError
            If name is empty or db_path is invalid
        IOError
            If metadata cannot be saved
        """
        # Input validation
        if not name or not name.strip():
            raise ValueError("Experiment name cannot be empty")
        if not db_path:
            raise ValueError("Database path must be provided")

        try:
            # Convert db_path to Path object and store as string
            db_path = Path(db_path)

            # Generate unique ID and timestamp
            experiment_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()

            # Store experiment metadata
            self.metadata["experiments"][experiment_id] = {
                "name": name.strip(),
                "timestamp": timestamp,
                "config": config,
                "db_path": str(db_path),
                "status": "registered",
            }

            self._save_metadata()
            logging.info(f"Registered new experiment: {name} (ID: {experiment_id})")
            return experiment_id

        except Exception as e:
            logging.error(f"Failed to register experiment '{name}': {e}")
            raise

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
        fill_method: str = "nan",
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple experiments with graceful handling of missing data.

        Parameters
        ----------
        experiment_ids : List[str]
            List of experiment IDs to compare
        metrics : Optional[List[str]]
            Specific metrics to compare. If None, compares all available metrics
        fill_method : str
            Method to handle missing values: 'nan', 'zero', or 'interpolate'

        Returns
        -------
        pd.DataFrame
            DataFrame containing the comparison data
        """
        # Validate inputs
        if not experiment_ids:
            raise ValueError("No experiment IDs provided")

        # Get all available metrics if none specified
        if metrics is None:
            metrics = self._get_all_available_metrics(experiment_ids)
            logging.info(f"Using all available metrics: {metrics}")

        # Initialize results storage
        results: Dict[str, Dict[str, Any]] = {}
        missing_metrics: Dict[str, List[str]] = {}

        # Collect data for each experiment
        for exp_id in experiment_ids:
            try:
                exp_data = self._get_experiment_metrics(exp_id, metrics)
                if exp_data.empty:
                    logging.warning(f"No metrics found for experiment '{exp_id}'")
                    continue

                results[exp_id] = exp_data

                # Track missing metrics
                available_metrics = set(exp_data.columns)
                missing = [m for m in metrics if m not in available_metrics]
                if missing:
                    missing_metrics[exp_id] = missing
                    logging.warning(
                        f"Experiment '{exp_id}' is missing metrics: {missing}"
                    )

            except Exception as e:
                logging.error(f"Error processing experiment '{exp_id}': {str(e)}")
                continue

        if not results:
            raise ValueError("No valid data found for any experiment")

        # Create combined DataFrame
        df = self._combine_experiment_data(results, fill_method)

        # Add metadata
        df = self._add_experiment_metadata(df, experiment_ids)

        return df

    def _get_all_available_metrics(self, experiment_ids: List[str]) -> List[str]:
        """Get union of all metrics available across experiments."""
        all_metrics = set()
        for exp_id in experiment_ids:
            try:
                metrics = self._get_experiment_metrics(exp_id).columns
                all_metrics.update(metrics)
            except Exception as e:
                logging.warning(
                    f"Could not get metrics for experiment '{exp_id}': {str(e)}"
                )
        return sorted(all_metrics)

    def _combine_experiment_data(
        self, results: Dict[str, pd.DataFrame], fill_method: str
    ) -> pd.DataFrame:
        """Combine experiment data with proper handling of missing values."""
        # Create list of DataFrames with experiment ID as index
        dfs = []
        for exp_id, data in results.items():
            df = data.copy()
            df["experiment_id"] = exp_id
            dfs.append(df)

        # Combine all DataFrames
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)

        # Handle missing values based on specified method
        if fill_method == "zero":
            combined_df.fillna(0, inplace=True)
        elif fill_method == "interpolate":
            combined_df = combined_df.groupby("experiment_id").apply(
                lambda x: x.interpolate(method="linear")
            )
        # 'nan' method leaves values as NaN

        return combined_df

    def _add_experiment_metadata(
        self, df: pd.DataFrame, experiment_ids: List[str]
    ) -> pd.DataFrame:
        """Add experiment metadata to the DataFrame."""
        metadata = []
        for exp_id in experiment_ids:
            if exp_id in self.metadata["experiments"]:
                exp_meta = self.metadata["experiments"][exp_id]
                metadata.append(
                    {
                        "experiment_id": exp_id,
                        "name": exp_meta.get("name", ""),
                        "description": exp_meta.get("description", ""),
                        "timestamp": exp_meta.get("timestamp", ""),
                        "config": str(exp_meta.get("config", {})),
                    }
                )

        meta_df = pd.DataFrame(metadata)
        return df.merge(meta_df, on="experiment_id", how="left")

    def generate_comparison_summary(
        self, experiment_ids: List[str], metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a summary of the comparison including missing data statistics.

        Parameters
        ----------
        experiment_ids : List[str]
            List of experiment IDs to compare
        metrics : Optional[List[str]]
            Specific metrics to compare

        Returns
        -------
        Dict[str, Any]
            Summary statistics and missing data information
        """
        df = self.compare_experiments(experiment_ids, metrics)

        summary = {
            "total_experiments": len(experiment_ids),
            "valid_experiments": df["experiment_id"].nunique(),
            "metrics_analyzed": list(df.select_dtypes(include=[np.number]).columns),
            "missing_data": {
                "total_missing_values": df.isna().sum().to_dict(),
                "missing_percentage": (df.isna().mean() * 100).round(2).to_dict(),
            },
            "basic_stats": df.describe().to_dict(),
            "warnings": [],
        }

        # Add warnings for significant missing data
        for col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            if missing_pct > 20:
                summary["warnings"].append(
                    f"Metric '{col}' is missing {missing_pct:.1f}% of values"
                )

        return summary

    def _create_visualizations(self, df: pd.DataFrame, plot_path: Path) -> None:
        """
        Create visualization plots for experiment metrics.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing experiment metrics with columns:
            step_number, metric_name, metric_value, experiment_name
        plot_path : Path
            Path where the plot should be saved

        Raises
        ------
        Exception
            If plot creation or saving fails
        """
        try:
            metrics = df["metric_name"].unique()
            num_metrics = len(metrics)
            fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))

            # Handle single metric case
            if num_metrics == 1:
                axes = [axes]

            for ax, metric in zip(axes, metrics):
                metric_data = df[df["metric_name"] == metric]
                sns.lineplot(
                    data=metric_data,
                    x="step_number",
                    y="metric_value",
                    hue="experiment_name",
                    ax=ax,
                )
                ax.set_title(f"{metric} Over Time")
                ax.set_xlabel("Simulation Step")
                ax.set_ylabel("Value")
                ax.legend(title="Experiment")

            plt.tight_layout()
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logging.debug(f"Created visualization plots at {plot_path}")

        except Exception as e:
            logging.error(f"Failed to create visualizations: {e}")
            raise

    def generate_comparison_report(
        self, experiment_ids: List[str], output_file: Path | str | None = None
    ) -> None:
        """
        Generate a comparison report for multiple experiments using HTML templates.

        Parameters
        ----------
        experiment_ids : List[str]
            List of experiment IDs to compare
        output_file : Path, optional
            Output file path for the report
        """
        # Prepare data for template
        experiment_data = {
            exp_id: {
                "name": self.metadata["experiments"][exp_id]["name"],
                "description": self.metadata["experiments"][exp_id]["description"],
                "timestamp": self.metadata["experiments"][exp_id]["timestamp"],
                "metrics": self._get_experiment_metrics(exp_id),
            }
            for exp_id in experiment_ids
        }

        # Generate config comparison table
        config_table = self._generate_config_comparison_table(experiment_ids)

        # Generate performance plots
        plot_path = self._generate_comparison_plots(experiment_ids)

        # Generate statistics summary
        stats_summary = self._generate_statistics_summary(experiment_ids)

        # Render template
        template = self.template_env.get_template("comparison_report.html")
        html = template.render(
            experiments=experiment_data,
            config_table=config_table,
            plot_path=plot_path.name if plot_path else None,
            stats_summary=stats_summary,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Write output
        if output_file is None:
            output_file = Path(f"comparison_report_{datetime.now():%Y%m%d_%H%M%S}.html")

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html)

    def _generate_config_comparison_table(self, experiment_ids: List[str]) -> str:
        """
        Generate an HTML table comparing configurations across experiments.

        Parameters
        ----------
        experiment_ids : List[str]
            List of experiment IDs to compare

        Returns
        -------
        str
            HTML string containing the configuration comparison table
        """
        # Flatten all configs
        configs = {
            exp_id: self._flatten_config(self.metadata["experiments"][exp_id]["config"])
            for exp_id in experiment_ids
        }

        # Get all unique parameters
        all_params = set()
        for config in configs.values():
            all_params.update(config.keys())

        # Generate header row
        header_cells = []
        for exp_id in experiment_ids:
            header_cells.append(
                f'<th>{self.metadata["experiments"][exp_id]["name"]}</th>'
            )

        # Start table
        table_parts = [
            "<table>",
            "<tr>",
            "<th>Parameter</th>",
            "".join(header_cells),
            "</tr>",
        ]

        # Generate rows for each parameter
        for param in sorted(all_params):
            values = [str(configs[exp_id].get(param, "")) for exp_id in experiment_ids]
            all_same = len(set(values)) == 1

            # Generate cells for this row
            cells = []
            for val in values:
                style = "" if all_same else ' style="background-color: #ffeb3b36"'
                cells.append(f"<td{style}>{val}</td>")

            # Add row to table
            table_parts.append(f"<tr><td>{param}</td>{''.join(cells)}</tr>")

        # Close table
        table_parts.append("</table>")

        return "\n".join(table_parts)

    def _generate_statistical_summary(self, df: pd.DataFrame) -> str:
        """
        Generate statistical summary of experimental results.

        Parameters
        ----------
            df (pd.DataFrame): DataFrame containing experiment results

        Returns
        -------
            str: HTML string containing the statistical summary table
        """
        summary = df.groupby(["experiment_name", "metric_name"])[
            "metric_value"
        ].describe()
        return f"<pre>{summary.to_string()}</pre>"

    def export_experiment_data(
        self, experiment_id: str, output_path: Path | str
    ) -> None:
        """
        Export experiment data to CSV.

        Parameters
        ----------
        experiment_id : str
            ID of the experiment to export
        output_path : Path | str
            Path where the CSV file should be saved

        Raises
        ------
        ValueError
            If experiment_id is not found
        sqlite3.Error
            If database operations fail
        IOError
            If CSV file cannot be written
        """
        output_path = Path(output_path)
        exp_data = self.metadata["experiments"].get(experiment_id)
        if not exp_data:
            raise ValueError(f"Experiment {experiment_id} not found")

        try:
            db_path = Path(exp_data["db_path"])
            with sqlite3.connect(db_path) as conn, output_path.open(
                "w", encoding="utf-8", newline=""
            ) as csv_file:

                # Create CSV writer
                writer = csv.writer(csv_file)

                # Write headers
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM SimulationMetrics LIMIT 1")
                headers = [description[0] for description in cursor.description]
                writer.writerow(headers)

                # Write data in chunks
                chunk_size = 1000
                for offset in range(0, self._get_row_count(cursor), chunk_size):
                    cursor.execute(
                        "SELECT * FROM SimulationMetrics LIMIT ? OFFSET ?",
                        (chunk_size, offset),
                    )
                    writer.writerows(cursor.fetchall())

                logging.info(
                    f"Successfully exported data for experiment {experiment_id}"
                )

        except sqlite3.Error as e:
            logging.error(f"Database error during export: {e}")
            raise
        except IOError as e:
            logging.error(f"File error during export: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during export: {e}")
            raise

    def _get_row_count(self, cursor: sqlite3.Cursor) -> int:
        """Helper method to get total row count."""
        cursor.execute("SELECT COUNT(*) FROM SimulationMetrics")
        return cursor.fetchone()[0]

    def cleanup_old_experiments(self, days_old: int = 30) -> None:
        """
        Remove experiments older than specified days.

        Parameters
        ----------
        days_old : int, optional
            Number of days after which experiments should be removed.
            Defaults to 30

        Raises
        ------
        IOError
            If files cannot be deleted
        Exception
            If cleanup operations fail
        """
        current_time = datetime.now(timezone.utc)
        experiments_to_remove = []

        try:
            for exp_id, exp_data in self.metadata["experiments"].items():
                exp_date = datetime.fromisoformat(exp_data["timestamp"])
                if (current_time - exp_date).days > days_old:
                    experiments_to_remove.append(exp_id)

            for exp_id in experiments_to_remove:
                exp_data = self.metadata["experiments"][exp_id]

                # Remove database file
                db_path = Path(exp_data["db_path"])
                if db_path.exists():
                    db_path.unlink()
                    logging.info(f"Removed database for experiment {exp_id}")

                # Remove from metadata
                del self.metadata["experiments"][exp_id]
                logging.info(f"Removed metadata for experiment {exp_id}")

            self._save_metadata()
            logging.info(f"Cleaned up {len(experiments_to_remove)} old experiments")

        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            raise
