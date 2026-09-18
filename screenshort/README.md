# Screenshot Test Evidence

This folder contains screenshots captured while manually testing the MemoryFAQ-RAG Streamlit application. They show the main user flows and expected behavior of the chat interface.

| Screenshot | Test scenario | Expected behavior shown |
| --- | --- | --- |
| [chatbot-home.png](chatbot-home.png) | Open the application | The chatbot landing page and its main interface load successfully. |
| [working-hours.png](working-hours.png) | Ask a question answered by the local knowledge base | The application returns a grounded answer for a working-hours query. |
| [remote-work.png](remote-work.png) | Ask another local-document question | The application retrieves and answers a remote-work policy query. |
| [web-rag.png](web-rag.png) | Query content added from a web page | The application responds using web-ingested knowledge, demonstrating the web RAG flow. |
| [unsupported-question.png](unsupported-question.png) | Ask a question outside the indexed knowledge | The application handles an unsupported question safely instead of presenting an ungrounded answer. |

These screenshots provide visual evidence that the interface, local-document retrieval, web-page retrieval, and unsupported-query handling were tested.
