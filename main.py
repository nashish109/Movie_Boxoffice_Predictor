import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def load_data(file_path):
    """
    Load movie data from a file (CSV, JSON, or Excel).

    Args:
        file_path (str): The path to the data file.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def perform_eda(df):
    """
    Perform exploratory data analysis on the movie dataset.

    Args:
        df (pandas.DataFrame): The movie data.
    """
    print("\nExploratory Data Analysis:")
    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nCategorical Feature Distribution:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n--- {col} ---")
        print(df[col].value_counts())

def clean_data(df):
    """
    Clean and preprocess the movie data.

    Args:
        df (pandas.DataFrame): The movie data.

    Returns:
        pandas.DataFrame: The cleaned data.
    """
    # Handle missing values
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Handle outliers using IQR method
    for col in ['budget', 'revenue']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Encode categorical features
    df = pd.get_dummies(df, columns=['genre'], drop_first=True)

    # Extract features from release_date
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day
    df.drop(columns=['release_date', 'cast', 'director'], inplace=True)

    # Log-transform skewed numerical features
    df['revenue'] = df['revenue'].apply(lambda x: np.log1p(x))
    
    return df

def train_model(df):
    """
    Train a machine learning model to predict box office revenue.

    Args:
        df (pandas.DataFrame): The cleaned movie data.

    Returns:
        tuple: A tuple containing the model, R-squared score, and MAE.
    """
    target = 'revenue'
    features = [col for col in df.columns if col != target]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_test) < 2:
        print("\nWarning: Test set is too small to calculate R-squared. Skipping model evaluation.")
        return None, float('nan'), float('nan')

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nModel Performance:")
    print(f"R-squared: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    return model, r2, mae

def generate_visualizations(df, original_df):
    """
    Generate and save visualizations for the movie data.

    Args:
        df (pandas.DataFrame): The cleaned movie data.
        original_df (pandas.DataFrame): The original movie data before cleaning.
    """
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig(os.path.join(plots_dir, f'correlation_heatmap_{timestamp}.png'))
    plt.close()

    # Bar plot of average revenue by genre
    plt.figure(figsize=(12, 6))
    original_df.groupby('genre')['revenue'].mean().sort_values(ascending=False).plot(kind='bar')
    plt.title('Average Revenue by Genre')
    plt.ylabel('Average Revenue')
    plt.savefig(os.path.join(plots_dir, f'average_revenue_by_genre_{timestamp}.png'))
    plt.close()

    # Scatter plot of budget vs. revenue
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=original_df, x='budget', y='revenue')
    plt.title('Budget vs. Revenue')
    plt.savefig(os.path.join(plots_dir, f'budget_vs_revenue_{timestamp}.png'))
    plt.close()

    # Distribution plot of revenue (log-transformed)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['revenue'], kde=True)
    plt.title('Distribution of Log-Transformed Revenue')
    plt.savefig(os.path.join(plots_dir, f'revenue_distribution_{timestamp}.png'))
    plt.close()

    print(f"\nVisualizations saved to '{plots_dir}' directory.")

def generate_report(eda_summary, r2, mae, model):
    """
    Generate a text report with EDA summary, model performance, and feature importance.

    Args:
        eda_summary (dict): A dictionary containing EDA summary.
        r2 (float): The R-squared score of the model.
        mae (float): The Mean Absolute Error of the model.
        model (sklearn.linear_model.LinearRegression): The trained model.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'box_office_report_{timestamp}.txt'

    with open(report_path, 'w') as f:
        f.write("Box Office Revenue Prediction Report\n")
        f.write("="*40 + "\n")
        f.write(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("1. Exploratory Data Analysis (EDA) Summary\n")
        f.write("-" * 40 + "\n")
        f.write("Descriptive Statistics:\n")
        f.write(str(eda_summary['descriptive_stats']) + "\n\n")
        f.write("Missing Values:\n")
        f.write(str(eda_summary['missing_values']) + "\n\n")

        f.write("2. Model Performance\n")
        f.write("-" * 40 + "\n")
        f.write(f"R-squared: {r2:.4f}\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n\n")

        f.write("3. Feature Importance\n")
        f.write("-" * 40 + "\n")
        feature_importance = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.coef_
        }).sort_values(by='importance', ascending=False)
        f.write(str(feature_importance) + "\n")

    print(f"\nReport saved to '{report_path}'")

def generate_dummy_data(num_rows=100):
    """Generate a dummy movie dataset."""
    genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Sci-Fi', 'Horror', 'Romance', 'Adventure']
    actors = [f'Actor {i}' for i in range(1, 21)]
    directors = [f'Director {i}' for i in range(1, 11)]

    data = {
        'budget': np.random.randint(1000000, 200000000, size=num_rows),
        'genre': np.random.choice(genres, size=num_rows),
        'runtime': np.random.randint(80, 180, size=num_rows),
        'release_date': [datetime.date(2020, 1, 1) + datetime.timedelta(days=np.random.randint(0, 365*3)) for _ in range(num_rows)],
        'cast': [', '.join(np.random.choice(actors, size=np.random.randint(2, 5), replace=False)) for _ in range(num_rows)],
        'director': np.random.choice(directors, size=num_rows),
        'imdb_rating': np.round(np.random.uniform(4.0, 9.0, size=num_rows), 1),
        'social_media_sentiment': np.round(np.random.uniform(0.2, 0.9, size=num_rows), 2),
        'revenue': np.random.randint(5000000, 800000000, size=num_rows)
    }
    return pd.DataFrame(data)


def main(file_path):
    """
    Main function to run the movie box office prediction pipeline.

    Args:
        file_path (str): The path to the data file.
    """
    try:
        movie_data = load_data(file_path)
        print("Data loaded successfully:")
        print(movie_data.head())

        # Perform EDA
        eda_summary = {
            'descriptive_stats': movie_data.describe(),
            'missing_values': movie_data.isnull().sum()
        }
        perform_eda(movie_data)

        # Clean the data
        cleaned_data = clean_data(movie_data.copy())
        print("\nCleaned Data:")
        print(cleaned_data.head())

        # Save cleaned data
        cleaned_data.to_csv('movies_cleaned.csv', index=False)
        print("\nCleaned data saved to 'movies_cleaned.csv'")

        # Train the model
        model, r2, mae = train_model(cleaned_data)

        if model:
            # Generate visualizations
            generate_visualizations(cleaned_data, movie_data)

            # Generate report
            generate_report(eda_summary, r2, mae, model)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    # Create a dummy csv for testing
    dummy_df = generate_dummy_data(100)
    dummy_df.to_csv('movies.csv', index=False)

    main('movies.csv')