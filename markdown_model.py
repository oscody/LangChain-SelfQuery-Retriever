from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.base import AttributeInfo, StructuredQuery
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSerializable
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from pinecone import Pinecone
import json
import os
from dotenv import load_dotenv
from typing import Optional, Dict

class MarkdownSearchModel:
    RETRIEVER_MODEL_NAME: str = None
    SUMMARY_MODEL_NAME: str = None
    EMBEDDING_MODEL_NAME: str = None
    constructor_prompt: Optional[ChatPromptTemplate] = None
    vectorstore: Optional[PineconeVectorStore] = None
    retriever: Optional[SelfQueryRetriever] = None
    query_constructor: RunnableSerializable[Dict, StructuredQuery] = None
    top_k: int = None

    def __init__(self, **kwargs):
        super().__init__()
        load_dotenv()
        with open('./config.json') as f:
            config = json.load(f)
            self.RETRIEVER_MODEL_NAME = config["RETRIEVER_MODEL_NAME"]
            self.SUMMARY_MODEL_NAME = config["SUMMARY_MODEL_NAME"]
            self.EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL_NAME"]
            self.top_k = config["top_k"]
        self.initialize_query_constructor()
        self.initialize_vector_store()
        self.initialize_retriever()
        self.initialize_chat_model(config)

    def initialize_query_constructor(self):
        document_content_description = "Markdown documents with extracted metadata."
        allowed_comparators = [
            "$eq",  # Equal to (number, string, boolean)
            "$ne",  # Not equal to (number, string, boolean)
            "$gt",  # Greater than (number)
            "$gte",  # Greater than or equal to (number)
            "$lt",  # Less than (number)
            "$lte",  # Less than or equal to (number)
            "$in",  # In array (string or number)
            "$nin",  # Not in array (string or number)
        ]
        allowed_operators = [
            "AND",
            "OR"
        ]

        metadata_field_info = [
            AttributeInfo(name="name", description="The name of the document", type="string"),
            AttributeInfo(name="tags", description="Tags associated with the document", type="list[string]"),  # Change type
            AttributeInfo(name="date-created", description="The date the document was created", type="date"),
        ]

        # self.constructor_prompt = get_query_constructor_prompt(
        #     document_content_description,
        #       metadata_field_info, 
        #       allowed_comparators, 
        #       allowed_operators
        # )

        self.constructor_prompt = get_query_constructor_prompt(
            document_content_description,
              metadata_field_info
        )

    def initialize_vector_store(self):
        PINECONE_KEY, PINECONE_INDEX_NAME = os.getenv(
            'PINECONE_API_KEY'), os.getenv('PINECONE_INDEX_NAME')
        pc = Pinecone(api_key=PINECONE_KEY)
        pc_index = pc.Index(PINECONE_INDEX_NAME)
        embeddings = OpenAIEmbeddings(model=self.EMBEDDING_MODEL_NAME)
        namespace = "document_search"
        self.vectorstore = PineconeVectorStore(
            index=pc_index, 
            embedding=embeddings, 
            namespace=namespace
            )

    def initialize_retriever(self):
        query_model = ChatOpenAI(
            model=self.RETRIEVER_MODEL_NAME,
              temperature=0, 
              streaming=True
              )
        output_parser = StructuredQueryOutputParser.from_components()
        self.query_constructor = self.constructor_prompt | query_model | output_parser
        self.retriever = SelfQueryRetriever(
            query_constructor=self.query_constructor,
            vectorstore=self.vectorstore,
            structured_query_translator=PineconeTranslator(),
            search_kwargs={'k': self.top_k}
        )

    def initialize_chat_model(self, config):
        def format_docs(docs):
            return "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)

        chat_model = ChatOpenAI(
            model=self.SUMMARY_MODEL_NAME,
            temperature=config['TEMPERATURE'],
            streaming=True,
            max_retries=10
        )

        prompt = ChatPromptTemplate.from_messages(
            [
            ('system', 
             """
                Your goal is to retrieve and answer questions based user queries.
                If a retrieved markdown documents doesn't seem relevant, omit it from your response. 
                If your context is empty or none of the retrieved markdown documents are relevant,
                tell the user you couldn't find any markdown documents that match their query.

                Question: {question}
                Context: {context}
                """
            )
        ]
    )

        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["context"]))) | prompt | chat_model | StrOutputParser()
        )

        self.rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough(), "query_constructor": self.query_constructor}
        ).assign(answer=rag_chain_from_docs)

    def search_documents(self, query: str):
        try:
            result = self.rag_chain_with_source.invoke(query)
            return {
                'answer': result['answer'],
                'context': "\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in result['context'])
            }
        except Exception as e:
            return {'error': f"An error occurred: {e}", 'context': ""}
