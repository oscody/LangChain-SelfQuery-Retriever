from markdown_model import MarkdownSearchModel


chat_model = MarkdownSearchModel()
query = "what notes do i have on self reflection has a tag named to do or AI Ideas"

query_constructor = chat_model.query_constructor.invoke(query)


print(f"query_constructor: {query_constructor}")

print("response:")
for chunk in chat_model.rag_chain_with_source.stream(query):
    print(chunk)
