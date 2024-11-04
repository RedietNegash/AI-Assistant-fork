import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    print(encoding)
    num_tokens = len(encoding.encode(string))
    print("num_tokens",num_tokens)
    return num_tokens

num_tokens_from_string("tiktoken is", "cl100k_base")