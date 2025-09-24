"""
Medical Knowledge Graph RAG System
Main module for setting up connections and RAG chains
"""

print("🚀 Starting Medical RAG System...")

import os
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import PromptTemplate

print("✓ Imports successful...")

# Load environment variables
load_dotenv()

class MedicalRAGSystem:
    """Main class for the Medical RAG System"""
    
    def __init__(self):
        self.graph = None
        self.llm = None
        self.graph_qa_chain = None
        self.optimized_chain = None
        self.optimized_chain_verbose = None
        
    def setup_connections(self):
        """Set up Neo4j and Azure OpenAI connections"""
        print("Setting up connections...")
        
        # Display environment variables for verification
        print("NEO4J_2_URI:", os.getenv("NEO4J_2_URI"))
        print("NEO4J_2_USERNAME:", os.getenv("NEO4J_2_USERNAME"))
        print("NEO4J_2_PASSWORD:", os.getenv("NEO4J_2_PASSWORD"))
        print("NEO4J_2_DATABASE:", os.getenv("NEO4J_2_DATABASE"))
        
        # Connect to Neo4j Environment 2 with proper database specification
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_2_URI"),
            username=os.getenv("NEO4J_2_USERNAME"),
            password=os.getenv("NEO4J_2_PASSWORD"),
            database=os.getenv("NEO4J_2_DATABASE")
        )
        
        # Test the connection with a simple query
        result = self.graph.query("RETURN 'Connection successful!' as message")
        print("Connection test result:", result)
        
        # Set up Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.1,
            max_tokens=1000
        )
        
        print("✓ Connections established successfully!")
        
    def setup_rag_system(self):
        """Set up RAG system for knowledge graph querying"""
        if not self.graph or not self.llm:
            raise ValueError("Connections must be established first. Call setup_connections().")
            
        try:
            # Try using the new GraphCypherQAChain from langchain_neo4j
            from langchain_neo4j import GraphCypherQAChain
            
            print("Setting up knowledge graph RAG system with langchain_neo4j...")
            
            # Set up GraphCypherQAChain for knowledge graph querying
            self.graph_qa_chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True,
                return_intermediate_steps=True,
                allow_dangerous_requests=True  # Required for security acknowledgment
            )
            
            print("✓ Graph QA chain setup complete!")
            
        except ImportError:
            print("langchain_neo4j GraphCypherQAChain not available, trying community version...")
            from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
            
            # Fallback to community version with proper error handling
            try:
                self.graph_qa_chain = GraphCypherQAChain.from_llm(
                    llm=self.llm,
                    graph=self.graph,
                    verbose=True,
                    return_intermediate_steps=True,
                    allow_dangerous_requests=True
                )
                print("✓ Community Graph QA chain setup complete!")
            except Exception as e:
                print(f"Error with community version: {e}")
                self._setup_simple_graph_qa()

        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Setting up simple function-based approach...")
            self._setup_simple_graph_qa()
            
    def _setup_simple_graph_qa(self):
        """Create a simple function-based approach as fallback"""
        def simple_graph_qa(question):
            # Get schema information
            schema_info = self.graph.get_schema
            
            # Create a simple prompt
            cypher_prompt = f"""
            Based on this Neo4j schema:
            {schema_info}
            
            Generate a Cypher query to answer: {question}
            
            Return only the Cypher query, no explanation.
            """
            
            # Get cypher query from LLM
            response = self.llm.invoke(cypher_prompt)
            cypher_query = response.content.strip()
            
            print(f"Generated Cypher: {cypher_query}")
            
            # Execute the query
            try:
                result = self.graph.query(cypher_query)
                return result
            except Exception as query_error:
                return f"Query execution error: {query_error}"
        
        # Store the function as our QA chain
        self.graph_qa_chain = simple_graph_qa
        print("✓ Simple function-based Graph QA setup complete!")
        
    def get_schema(self):
        """Get the database schema"""
        if not self.graph:
            raise ValueError("Graph connection not established. Call setup_connections() first.")
        return self.graph.schema
        
    def test_basic_query(self, question="What are the symptoms of diabetes?"):
        """Test the GraphCypherQAChain with a sample query"""
        if not self.graph_qa_chain:
            raise ValueError("RAG system not set up. Call setup_rag_system() first.")
            
        print(f"Testing with question: {question}")
        print("-" * 50)
        
        result = self.graph_qa_chain.invoke({"query": question})
        print("Result:", result)
        return result

    def initialize_system(self):
        """Initialize the complete system"""
        print("🚀 Initializing Medical RAG System...")
        self.setup_connections()
        self.setup_rag_system()
        print("✅ System initialization complete!")
        return self


def main():
    """Main function to demonstrate system setup"""
    from utils import create_optimized_chain
    
    # Create and initialize the system
    medical_rag = MedicalRAGSystem()
    medical_rag.initialize_system()
    
    # Display schema
    print("\n📊 Database Schema:")
    print(medical_rag.get_schema())
    
    # Create optimized chain with better templates
    print("\n🔧 Setting up optimized query chain...")
    optimized_chain = create_optimized_chain(medical_rag.llm, medical_rag.graph)
    
    # Let's first check what diseases are available in the database
    print("\n🔍 Checking available diseases (first 10)...")
    try:
        diseases = medical_rag.graph.query("MATCH (d:Disease) RETURN d.name as disease_name LIMIT 10")
        print("Available diseases:")
        for disease in diseases:
            print(f"  - {disease['disease_name']}")
    except Exception as e:
        print(f"Error checking diseases: {e}")
    
    # Start continuous chat loop
    print("\n🩺 Medical RAG Chat System")
    print("=" * 50)
    print("Ask medical questions and get answers from the knowledge graph.")
    print("Press Ctrl+C to exit the chat.")
    print("⚠️  Always consult healthcare professionals for medical advice.")
    print("=" * 50)
    
    try:
        while True:
            print("\n" + "-" * 30)
            user_question = input("🧪 Enter your medical question: ").strip()
            
            if not user_question:
                print("❌ Please enter a valid question.")
                continue
            
            print(f"\n🤖 Processing: {user_question}")
            print("-" * 50)
            
            try:
                if optimized_chain.is_available():
                    optimized_chain.ask_medical_question_clean(user_question, show_cypher=True)
                else:
                    medical_rag.test_basic_query(user_question)
            except Exception as e:
                print(f"❌ Error processing question: {str(e)}")
            
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using the Medical RAG System!")
        print("Stay healthy and always consult healthcare professionals!")
    
    return medical_rag


if __name__ == "__main__":
    system = main()