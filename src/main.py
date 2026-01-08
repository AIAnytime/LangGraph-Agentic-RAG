from langchain_core.messages import HumanMessage
from .retriever import ingest_documents, get_retriever_tool
from .agents.graph import build_graph

def main():
    print("🚀 Initializing Agentic RAG System...")
    
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
    ]
    
    print("\n📚 Ingesting documents into FAISS vector store...")
    try:
        vectorstore = ingest_documents(urls)
        print("✅ Documents ingested successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not ingest documents: {e}")
        import traceback
        traceback.print_exc()
        return
    
    retriever_tool = get_retriever_tool(vectorstore)
    tools = [retriever_tool]
    
    print("\n🔧 Building LangGraph state machine...")
    app = build_graph(tools)
    print("✅ Graph built successfully!")
    
    print("\n" + "="*60)
    print("🤖 Agentic RAG System Ready!")
    print("="*60)
    
    questions = [
        "What are the key components of an AI agent?",
        "How does prompt engineering improve LLM performance?"
    ]
    
    for question in questions:
        print(f"\n\n{'='*60}")
        print(f"❓ Question: {question}")
        print('='*60)
        
        inputs = {"messages": [HumanMessage(content=question)]}
        
        print("\n🔄 Processing...\n")
        
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"📍 Node: {key}")
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        print(f"💬 Output: {last_msg.content[:200]}...")
                    elif hasattr(last_msg, 'tool_calls'):
                        print(f"🔧 Tool Call: {last_msg.tool_calls}")
                print()
        
        final_message = output[list(output.keys())[-1]]["messages"][-1]
        print("\n" + "="*60)
        print("✨ FINAL ANSWER:")
        print("="*60)
        print(final_message.content)
        print("="*60)

if __name__ == "__main__":
    main()
