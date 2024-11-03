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
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

    Methods
    -------
    register_experiment(self, name: str, config: Dict[str, Any], db_path: Path | str) -> str:
        Register a new experiment run with its configuration and database location.
    compare_experiments(self, experiment_ids: List[str], metrics: List[str] = None) -> pd.DataFrame:
        Compare results from multiple experiments by retrieving specified metrics.
    generate_comparison_report(self, experiment_ids: List[str], output_file: Path | str | None = None):
        Generate a comprehensive HTML report comparing multiple experiments.
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
        self, experiment_ids: List[str], metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare results from multiple experiments by retrieving specified metrics.

        Parameters
        ----------
        experiment_ids : List[str]
            List of experiment IDs to compare
        metrics : List[str], optional
            List of metric names to compare. Defaults to
            ["total_agents", "total_resources", "average_agent_resources"]

        Returns
        -------
        pd.DataFrame
            DataFrame containing metrics data for all experiments,
            with columns for step_number, metric_name, metric_value,
            experiment_id, and experiment_name

        Raises
        ------
        sqlite3.Error
            If database operations fail
        pd.io.sql.DatabaseError
            If data cannot be read into DataFrame
        """
        if metrics is None:
            metrics = ["total_agents", "total_resources", "average_agent_resources"]

        results = []
        for exp_id in experiment_ids:
            exp_data = self.metadata["experiments"].get(exp_id)
            if exp_data is None:
                logging.warning(f"Experiment ID '{exp_id}' not found in metadata.")
                continue

            try:
                with sqlite3.connect(exp_data["db_path"]) as conn:
                    placeholders = ",".join("?" * len(metrics))
                    query = f"""
                        SELECT s.step_number, m.metric_name, m.metric_value
                        FROM SimulationMetrics m
                        JOIN SimulationSteps s ON s.step_id = m.step_id
                        WHERE m.metric_name IN ({placeholders})
                        ORDER BY s.step_number
                    """
                    df = pd.read_sql_query(query, conn, params=metrics)
                    df["experiment_id"] = exp_id
                    df["experiment_name"] = exp_data["name"]
                    results.append(df)
            except sqlite3.Error as e:
                logging.error(f"Database error for experiment '{exp_id}': {e}")
                continue
            except pd.io.sql.DatabaseError as e:
                logging.error(f"Failed to read data for experiment '{exp_id}': {e}")
                continue

        if not results:
            logging.warning("No data was retrieved from any experiments.")
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

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
        Generate a comprehensive HTML report comparing multiple experiments.

        Parameters
        ----------
        experiment_ids : List[str]
            List of experiment IDs to include in the report
        output_file : Path | str | None, optional
            Path where the HTML report should be saved. If None, saves to
            experiments_dir/comparison_report.html

        Raises
        ------
        ValueError
            If no valid experiments are found
        IOError
            If report cannot be written to disk
        """
        try:
            # Setup output path
            if output_file is None:
                output_file = self.experiments_dir / "comparison_report.html"
            else:
                output_file = Path(output_file)

            # Get comparison data
            df = self.compare_experiments(experiment_ids)
            if df.empty:
                logging.warning("No data available for comparison report")
                return

            # Create visualizations
            plot_path = self.experiments_dir / "comparison_plots.png"
            self._create_visualizations(df, plot_path)

            # Generate HTML report
            html_content = self._generate_html_report(
                experiment_ids=experiment_ids, plot_path=plot_path, comparison_data=df
            )

            # Save report
            with output_file.open("w", encoding="utf-8") as f:
                f.write(html_content)

            logging.info(f"Generated comparison report at {output_file}")

        except Exception as e:
            logging.error(f"Failed to generate comparison report: {e}")
            raise

    def _generate_html_report(
        self, experiment_ids: List[str], plot_path: Path, comparison_data: pd.DataFrame
    ) -> str:
        """
        Generate HTML content for the comparison report.

        Parameters
        ----------
        experiment_ids : List[str]
            List of experiment IDs to include
        plot_path : Path
            Path to the generated visualization plots
        comparison_data : pd.DataFrame
            DataFrame containing comparison metrics

        Returns
        -------
        str
            Complete HTML content for the report

        Raises
        ------
        KeyError
            If required metadata is missing
        Exception
            If HTML generation fails
        """
        try:
            html = f"""
            <html>
            <head>
                <title>Experiment Comparison Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f5f5f5; }}
                    h1, h2 {{ color: #333; }}
                    .plot-container {{ margin: 20px 0; }}
                    .plot-container img {{ max-width: 100%; }}
                    .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Experiment Comparison Report</h1>
                
                <h2>Experiments Included</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Timestamp</th>
                    </tr>
                    {''.join(
                        f"<tr><td>{exp_id}</td>"
                        f"<td>{self.metadata['experiments'][exp_id]['name']}</td>"
                        f"<td>{self.metadata['experiments'][exp_id]['timestamp']}</td></tr>"
                        for exp_id in experiment_ids
                    )}
                </table>
                
                <h2>Configuration Comparison</h2>
                {self._generate_config_comparison_table(experiment_ids)}
                
                <h2>Results Visualization</h2>
                <div class="plot-container">
                    <img src="{plot_path}" alt="Experiment Comparison Plots"/>
                </div>
                
                <h2>Statistical Summary</h2>
                <div class="summary">
                    {self._generate_statistical_summary(comparison_data)}
                </div>
            </body>
            </html>
            """
            return html

        except Exception as e:
            logging.error(f"Failed to generate HTML content: {e}")
            raise

    def _generate_config_comparison_table(self, experiment_ids: List[str]) -> str:
        """
        Generate an HTML table comparing configurations across experiments.

        Parameters
        ----------
            experiment_ids (List[str]): List of experiment IDs to compare

        Returns
        -------
            str: HTML string containing the configuration comparison table
        """
        configs = {
            exp_id: self.metadata["experiments"][exp_id]["config"]
            for exp_id in experiment_ids
        }

        # Get all unique parameters
        all_params = set()
        for config in configs.values():
            all_params.update(config.keys())

        # Generate table
        html = "<table><tr><th>Parameter</th>"
        for exp_id in experiment_ids:
            html += f'<th>{self.metadata["experiments"][exp_id]["name"]}</th>'
        html += "</tr>"

        for param in sorted(all_params):
            html += f"<tr><td>{param}</td>"
            for exp_id in experiment_ids:
                value = configs[exp_id].get(param, "")
                html += f"<td>{value}</td>"
            html += "</tr>"

        html += "</table>"
        return html

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
