from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain import hub
import langchain

langchain.verbose = True
langchain.debug=True

def create_model(vectordb):
    
    with open('prompt.txt', 'r', encoding='utf-8') as file:
        prompt = file.read()

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    retrieval_qa_chat_prompt.messages[0].prompt.template = prompt

    llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                     openai_api_key='OPENAI_API_KEY',
                     temperature=0.6,
                     verbose=True,
                     )
    retriever= vectordb.as_retriever()

    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )

    chain = create_retrieval_chain(retriever, combine_docs_chain)

    return chain
