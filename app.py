from markdown_model import MarkdownSearchModel


chat_model = MarkdownSearchModel()
query = "Find notes on self-reflection where tag in ['to do', 'AI Ideas']"

query_constructor = chat_model.query_constructor.invoke(query)


print(f"query_constructor: {query_constructor}")

print("response:")
for chunk in chat_model.rag_chain_with_source.stream(query):
    print(chunk)
