import argparse
import gdown
import zipfile

# Specify the urls to download the weights from
t5_url = 'https://drive.google.com/file/d/1CkDT0MHauG9aGWWjcg2nihJzlDh9LJ9d/view?usp=sharing'
transformer_url = 'https://drive.google.com/file/d/1Td0LsCv8_1J-QgE_Me9Kj70HLZRqnWeb/view?usp=sharing'

# Specify the paths to save the downloaded weights to
t5_extract_path = './models/t5/'
t5_zip_path = t5_extract_path  + 'best.zip'
transformer_extract_path = './models/'
transfromer_zip_path = transformer_extract_path + 'pytorch_transformer.zip'
    
    
def unzip_weights(zip_path, extract_path):
    """
    Extracts the contents of a zip file to a specified directory.

    Args:
        zip_path: The path to the zip file to extract.
        extract_path: The path where the zip file contents should be extracted.
    """

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        
        
def download_weights(url, output, model='t5'):
    """
    Download weights for model from the given url and saves them to the specified output directory.
    
    Args:
        url: The url to download the weights from.
        output: The directory to save the downloaded weights to.
        model: The model name to download weights for (default 't5').
    """
    gdown.download(url, output, quiet=False, fuzzy=True)   
    
    if model == 't5':
        t5_zip_path = output + 'best.zip'
        unzip_weights(t5_zip_path, output) 
        
    if model == 'transformer':
        transfromer_zip_path = output + 'pytorch_transformer.zip'
        unzip_weights(transfromer_zip_path, output)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download weights for models.')
    parser.add_argument('--model', type=str, default='all', help='model to download weights for')
    args = parser.parse_args()
    
    if args.model == 't5':
        download_weights(t5_url, t5_extract_path, model='t5')
        
    elif args.model == 'transformer':
        download_weights(transformer_url, transformer_extract_path, model='transformer')
        
    else:
        download_weights(t5_url, t5_extract_path, model='t5')
        download_weights(transformer_url, transformer_extract_path, model='transformer')
        
    print('Done successfully! Check models folder.')
