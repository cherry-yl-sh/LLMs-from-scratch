import tiktoken
from importlib.metadata import version
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Poor Jack! It had always been his fate to have women say such things of him: the fact should be set down in extenuation. What struck me now was that, for the first time, he resented the tone. I had seen him, so often, basking under similar tributes--was it the conjugal note that robbed them of their savour? No--for, oddly enough, it became apparent that he was fond of Mrs. Gisburn--fond enough not to see her absurdity. It was his own absurdity he seemed to be wincing under--his own attitude as an object for garlands and incense."
    "of someunknownPlace.    "
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print("encode result " ,integers)

print("decode result ", tokenizer.decode(integers))