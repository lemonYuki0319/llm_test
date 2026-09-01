from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Tongyi
from config.settings import DASHSCOPE_API_KEY


class RAGHelper:
    def __init__(self, vectorstore_path="knowledge_base"):
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=DASHSCOPE_API_KEY
        )
        self.vectorstore_path = vectorstore_path
        self._vectorstore = None
        self._qa_chain = None

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = FAISS.load_local(
                self.vectorstore_path, self.embeddings,
                allow_dangerous_deserialization=True
            )
        return self._vectorstore

    @property
    def qa_chain(self):
        if self._qa_chain is None:
            llm = Tongyi(dashscope_api_key=DASHSCOPE_API_KEY)
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
        return self._qa_chain

    def search(self, query, k=3):
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

    def ask(self, question):
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result.get("source_documents", [])
        }

    def add_document(self, text, metadata=None):
        self.vectorstore.add_texts([text], metadatas=[metadata or {}])
        self.vectorstore.save_local(self.vectorstore_path)

    def rebuild_index(self):
        self._vectorstore = None
        self._qa_chain = None
