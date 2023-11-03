import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
import argparse


def unzip_ds(zip_path='./data/raw/filtered_paranmt.zip', 
             extract_path='./data/raw/'):
    """
    Extracts the contents of a zip file to a specified directory.

    Args:
        zip_path (str): The path to the zip file to extract.
        extract_path (str): The path where the zip file contents should be extracted.

    Returns:
        None
    """

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)



def load_df(ds_path= './data/raw/filtered.tsv', 
            delimiter='\t'):
    """
    Load the dataset into a DataFrame and remove the redundant column.
    
    Args:
        ds_path (str): path to the dataset file
        delimiter (str, optional): delimiter used in the format of dataset ('\t').
    
    Returns:
        DataFrame: loaded data
    """

    df = pd.read_csv(ds_path, delimiter=delimiter)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    return df


def preprocess(df):
    """
    Preprocess the original dataframe.
    
    Args:
        df (DataFrame): input dataframe.
        
    Returns:
        DataFrame: preprocessed data
    """
    for i, row in df.iterrows():
        if row['ref_tox'] < row['trn_tox']:
            df.at[i, 'reference'], df.at[i, 'translation'] = row['translation'], row['reference']
            df.at[i, 'ref_tox'], df.at[i, 'trn_tox'] = row['trn_tox'], row['ref_tox']
        
    df.rename(columns={'translation': 'detox_reference',
            'trn_tox': 'detox_ref_tox'}, 
            inplace=True)

    return df


def filter(df):
    """
    Filter the dataframe.
    
    Args:
        df (DataFrame): input dataframe.
    
    Returns:
        DataFrame: filtered data
    """
    filtered_df = df.query('detox_ref_tox <= 0.1 and ref_tox >= 0.9')
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def cut(df, size=10000):
    """
    Sort by reference's toxicity level in descending order and then cut the dataframe to a given size.
    
    Args:
        df (DataFrame): input dataframe.
        size (int): specified size.
    
    Returns:
        DataFrame: cutted data
    """
    if size != 424347:
        df.sort_values(by='ref_tox', ascending=False, inplace=True, ignore_index=True)
    return df[:size]
    

def retrieve_source_target(df, source='reference', target='detox_reference'):
    """
    Retrieve the specified columns (source and target) from a dataframe.
    
    Args:
        df (DataFrame): input dataframe.
        source (str): name of the source column to retrieve ('reference').
        target (str): name of the target column to retrieve ('detox_reference').
    
    Returns:
        DataFrame: data with source and target columns
    """
    return df[[source, target]]


def test_train_split(df, test_size=5000, random_state=420):
    """
    Split the input dataframe into training and testing sets.

    Args:
        df (pandas.DataFrame): The input dataframe to be split.
        test_size (float): The ratio of the test split.
        random_state (int): A random state.

    Returns:
        tuple: A tuple containing the training and testing dataframes
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train, df_test

def make_dataset():
    parser = argparse.ArgumentParser(description='Make a dataset.')
    
    # Default size is the full dataset size
    parser.add_argument('--size', type=int, default=424347, help='Size of the dataset to cut to.')
    args = parser.parse_args()

    # Extract data from zip file
    unzip_ds()

    # Load data into dataframe
    df = load_df()

    # Preprocess data
    df = preprocess(df)

    # Filter data
    df = filter(df)

    # Cut data to a specified size
    df = cut(df, size=args.size)

    # Retrieve source and target columns
    df = retrieve_source_target(df)

    # Save all data to CSV files
    df.to_csv('./data/interim/df.csv', index=False)
    
    # Split data into training and testing sets
    df_train, df_test = test_train_split(df)
    
    # Save train and test data to CSV files
    df_train.to_csv('./data/interim/train.csv', index=False)
    df_test.to_csv('./data/interim/test.csv', index=False)
    


if __name__ == '__main__':
    make_dataset()
    print('Done successfully! Check data/interim folder.')
