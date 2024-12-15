import pandas as pd
import matplotlib.pyplot as plt

# Define the analysis functions

def plot_population_dynamics(dataframe):
    """Plot total agents, system agents, and independent agents over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['step_number'], dataframe['total_agents'], label='Total Agents', color='blue')
    plt.plot(dataframe['step_number'], dataframe['system_agents'], label='System Agents', color='green')
    plt.plot(dataframe['step_number'], dataframe['independent_agents'], label='Independent Agents', color='orange')
    plt.plot(dataframe['step_number'], dataframe['control_agents'], label='Control Agents', color='red')
    plt.title('Population Dynamics Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Number of Agents')
    plt.legend()
    plt.show()

def plot_births_and_deaths(dataframe):
    """Plot births and deaths over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['step_number'], dataframe['births'], label='Births', color='green')
    plt.plot(dataframe['step_number'], dataframe['deaths'], label='Deaths', color='red')
    plt.title('Births and Deaths Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def plot_resource_efficiency(dataframe):
    """Plot resource efficiency and total resources over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['step_number'], dataframe['resource_efficiency'], label='Resource Efficiency', color='purple')
    plt.plot(dataframe['step_number'], dataframe['total_resources'], label='Total Resources', color='blue', linestyle='--')
    plt.title('Resource Efficiency and Total Resources Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_agent_health_and_age(dataframe):
    """Plot average agent health and age over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['step_number'], dataframe['average_agent_health'], label='Average Health', color='cyan')
    plt.plot(dataframe['step_number'], dataframe['average_agent_age'], label='Average Age', color='magenta')
    plt.title('Agent Health and Age Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_combat_metrics(dataframe):
    """Plot combat encounters and successful attacks over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['step_number'], dataframe['combat_encounters'], label='Combat Encounters', color='orange')
    plt.plot(dataframe['step_number'], dataframe['successful_attacks'], label='Successful Attacks', color='green')
    plt.title('Combat Metrics Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def plot_resource_sharing(dataframe):
    """Plot the amount of resources shared over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['step_number'], dataframe['resources_shared'], label='Resources Shared', color='gold')
    plt.title('Resources Shared Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Resources Shared')
    plt.legend()
    plt.show()

def plot_evolutionary_metrics(dataframe):
    """Plot genetic diversity and dominant genome ratio over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['step_number'], dataframe['genetic_diversity'], label='Genetic Diversity', color='blue')
    plt.plot(dataframe['step_number'], dataframe['dominant_genome_ratio'], label='Dominant Genome Ratio', color='purple')
    plt.title('Evolutionary Metrics Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_resource_distribution_entropy(dataframe):
    """Plot resource distribution entropy over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['step_number'], dataframe['resource_distribution_entropy'], label='Resource Distribution Entropy', color='darkred')
    plt.title('Resource Distribution Entropy Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Entropy')
    plt.legend()
    plt.show()

def plot_rewards(dataframe):
    """Plot average reward over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe['step_number'], dataframe['average_reward'], label='Average Reward', color='teal')
    plt.title('Average Reward Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()

def plot_allele_frequency_changes(dataframe):
    """Plot allele frequency changes over time."""
    plt.figure(figsize=(10, 6))
    allele_frequencies = dataframe.filter(like='allele_frequency_')
    for column in allele_frequencies.columns:
        plt.plot(dataframe['step_number'], dataframe[column], label=column)
    plt.title('Allele Frequency Changes Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Allele Frequency')
    plt.legend()
    plt.show()

# Load the dataset
def main(dataframe):

    try:

        # Call each function to analyze and visualize
        print("Plotting population dynamics...")
        plot_population_dynamics(dataframe)

        print("Plotting births and deaths...")
        plot_births_and_deaths(dataframe)

        print("Plotting resource efficiency...")
        plot_resource_efficiency(dataframe)

        print("Plotting agent health and age...")
        plot_agent_health_and_age(dataframe)

        print("Plotting combat metrics...")
        plot_combat_metrics(dataframe)

        print("Plotting resource sharing...")
        plot_resource_sharing(dataframe)

        print("Plotting evolutionary metrics...")
        plot_evolutionary_metrics(dataframe)

        print("Plotting resource distribution entropy...")
        plot_resource_distribution_entropy(dataframe)

        print("Plotting average rewards...")
        plot_rewards(dataframe)

        print("Plotting allele frequency changes...")
        plot_allele_frequency_changes(dataframe)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the analysis
if __name__ == "__main__":
    import pandas as pd
    from sqlalchemy import create_engine, inspect

    # connection_string = "sqlite:///simulations/simulation_20241110_122335.db"
    connection_string = "sqlite:///simulations/simulation.db"

    # Create engine
    engine = create_engine(connection_string)
    
    df = pd.read_sql("SELECT * FROM SimulationSteps", engine)

    main(df)
