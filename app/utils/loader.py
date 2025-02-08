from langchain_community.document_loaders import PyPDFLoader

def load(file_path: str):
    """
    Load a PDF file and return its content as a list of strings.
    
    Args:
        file_path (str): Path to the PDF file.
        
    Returns:
        list: List of documents (pages).
    """
    loader = PyPDFLoader(file_path)
    return loader.load()