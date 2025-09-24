"""
Utility functions and templates for Medical RAG System
Contains helper functions for medical queries and prompt templates
"""

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain


# Advanced Cypher Generation Template with Query Optimization
ADVANCED_CYPHER_TEMPLATE = """You are an expert Neo4j Cypher query generator for a medical knowledge graph.

DATABASE SCHEMA:
{schema}

MEDICAL KNOWLEDGE GRAPH STRUCTURE:
- Nodes: Disease, Symptom, Treatment, DiseaseSymptomRelationship, DiseaseTreatmentRelationship
- Properties: All nodes have "name" property containing the text value
- Key Relationships:
  * (DiseaseSymptomRelationship)-[:FOR_DISEASE]->(Disease)
  * (DiseaseSymptomRelationship)-[:FOR_SYMPTOM]->(Symptom)
  * (DiseaseTreatmentRelationship)-[:FOR_DISEASE]->(Disease)
  * (DiseaseTreatmentRelationship)-[:FOR_TREATMENT]->(Treatment)

CRITICAL RULES:
1. Use "name" property for all nodes, NOT "id" 
2. Use toLower() and CONTAINS for flexible matching
3. Use collect() to group results
4. Always add LIMIT for multiple results
5. Follow the relationship pattern through intermediate nodes

QUERY PATTERNS:

For symptoms questions ("What are symptoms of X", "X symptoms", "signs of X"):
MATCH (dsr:DiseaseSymptomRelationship)-[:FOR_DISEASE]->(d:Disease), 
      (dsr)-[:FOR_SYMPTOM]->(s:Symptom)
WHERE toLower(d.name) CONTAINS toLower("condition_name")
RETURN d.name as condition, collect(s.name) as symptoms
LIMIT 5

For treatment questions ("How to treat X", "treatment for X", "cure for X"):
MATCH (dtr:DiseaseTreatmentRelationship)-[:FOR_DISEASE]->(d:Disease), 
      (dtr)-[:FOR_TREATMENT]->(t:Treatment)
WHERE toLower(d.name) CONTAINS toLower("condition_name")
RETURN d.name as condition, collect(t.name) as treatments
LIMIT 5

For diagnostic questions ("What could cause X", "diseases with X symptom", "conditions with X"):
MATCH (dsr:DiseaseSymptomRelationship)-[:FOR_DISEASE]->(d:Disease), 
      (dsr)-[:FOR_SYMPTOM]->(s:Symptom)
WHERE toLower(s.name) CONTAINS toLower("symptom_name")
RETURN d.name as condition, count(s) as symptom_matches
ORDER BY symptom_matches DESC
LIMIT 10

For general condition info ("Tell me about X", "What is X"):
MATCH (d:Disease)
WHERE toLower(d.name) CONTAINS toLower("condition_name")
OPTIONAL MATCH (dsr:DiseaseSymptomRelationship)-[:FOR_DISEASE]->(d), (dsr)-[:FOR_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (dtr:DiseaseTreatmentRelationship)-[:FOR_DISEASE]->(d), (dtr)-[:FOR_TREATMENT]->(t:Treatment)
RETURN d.name as condition, collect(DISTINCT s.name) as symptoms, collect(DISTINCT t.name) as treatments
LIMIT 5

Question: {question}

Generate ONLY the Cypher query without any explanation:"""

# Create an improved QA prompt template for better responses
QA_TEMPLATE = """You are a helpful medical assistant. Based on the following information from a medical knowledge graph, provide a clear and helpful response to the patient's question.

Context from knowledge graph:
{context}

Question: {question}

Please provide a response as if you are talking to a patient. Be empathetic, clear, and always remind them to consult with healthcare professionals for proper diagnosis and treatment.

Answer:"""


class MedicalQueryHelper:
    """Helper class for medical database queries"""
    
    def __init__(self, graph):
        self.graph = graph
        
    def get_symptoms_for_condition(self, condition_name):
        """Get all symptoms for a specific medical condition"""
        query = f"""
        MATCH (dsr:DiseaseSymptomRelationship)-[:FOR_DISEASE]->(d:Disease), 
              (dsr)-[:FOR_SYMPTOM]->(s:Symptom)
        WHERE toLower(d.name) CONTAINS toLower('{condition_name}')
        RETURN d.name as condition, collect(s.name) as symptoms
        """
        result = self.graph.query(query)
        return result

    def get_treatments_for_condition(self, condition_name):
        """Get all treatments for a specific medical condition"""
        query = f"""
        MATCH (dtr:DiseaseTreatmentRelationship)-[:FOR_DISEASE]->(d:Disease), 
              (dtr)-[:FOR_TREATMENT]->(t:Treatment)
        WHERE toLower(d.name) CONTAINS toLower('{condition_name}')
        RETURN d.name as condition, collect(t.name) as treatments
        """
        result = self.graph.query(query)
        return result

    def find_conditions_by_symptoms(self, symptoms_list):
        """Find medical conditions that have any of the specified symptoms"""
        query = f"""
        MATCH (dsr:DiseaseSymptomRelationship)-[:FOR_DISEASE]->(d:Disease), 
              (dsr)-[:FOR_SYMPTOM]->(s:Symptom)
        WHERE toLower(s.name) IN [{', '.join([f"'{symptom.lower()}'" for symptom in symptoms_list])}]
        RETURN d.name as condition, collect(s.name) as matching_symptoms, count(s) as symptom_count
        ORDER BY symptom_count DESC
        LIMIT 10
        """
        result = self.graph.query(query)
        return result


class OptimizedMedicalChain:
    """Optimized chain setup for medical queries"""
    
    def __init__(self, llm, graph):
        self.llm = llm
        self.graph = graph
        self.optimized_chain = None
        self.optimized_chain_verbose = None
        self._setup_chains()
        
    def _setup_chains(self):
        """Set up optimized chains with improved templates"""
        try:
            # Create a clean version without verbose output
            self.optimized_chain = GraphCypherQAChain.from_llm(
                self.llm,
                graph=self.graph,
                verbose=False,  # Turn off verbose to avoid duplicate Cypher output
                allow_dangerous_requests=True,
                cypher_prompt=PromptTemplate(
                    input_variables=["schema", "question"], 
                    template=ADVANCED_CYPHER_TEMPLATE
                ),
                qa_prompt=PromptTemplate(
                    input_variables=["context", "question"],
                    template=QA_TEMPLATE
                ),
                return_intermediate_steps=True
            )

            # Also create a verbose version for debugging when needed
            self.optimized_chain_verbose = GraphCypherQAChain.from_llm(
                self.llm,
                graph=self.graph,
                verbose=True,  # Keep verbose for debugging
                allow_dangerous_requests=True,
                cypher_prompt=PromptTemplate(
                    input_variables=["schema", "question"], 
                    template=ADVANCED_CYPHER_TEMPLATE
                ),
                qa_prompt=PromptTemplate(
                    input_variables=["context", "question"],
                    template=QA_TEMPLATE
                ),
                return_intermediate_steps=True
            )
            
            print("✓ Advanced templates and chains created!")
            
        except Exception as e:
            print(f"Error setting up optimized chains: {e}")
            self.optimized_chain = None
            self.optimized_chain_verbose = None

    def ask_medical_question_clean(self, question, show_cypher=True):
        """Ask a medical question with clean output"""
        if not self.optimized_chain:
            raise ValueError("Optimized chains not available. Check setup.")
            
        # Always use verbose chain to show the green query building process
        if self.optimized_chain_verbose:
            result = self.optimized_chain_verbose.invoke({"query": question})
        else:
            result = self.optimized_chain.invoke({"query": question})
        
        # Always show the final answer regardless of which chain was used
        print("\n" + "="*60)
        print("🩺 MEDICAL ASSISTANT RESPONSE")
        print("="*60)
        print(f"📋 Question: {question}")
        print(f"💡 Answer: {result['result']}")
        print("="*60)
        print("⚠️  Always consult healthcare professionals for medical advice.")
        print("="*60)
        
        return result

    def is_available(self):
        """Check if optimized chains are available"""
        return self.optimized_chain is not None


def create_medical_query_helper(graph):
    """Factory function to create MedicalQueryHelper"""
    helper = MedicalQueryHelper(graph)
    print("✓ Helper functions created for medical queries!")
    return helper


def create_optimized_chain(llm, graph):
    """Factory function to create OptimizedMedicalChain"""
    chain = OptimizedMedicalChain(llm, graph)
    if chain.is_available():
        print("🚀 Improved medical AI assistant chain created!")
    else:
        print("⚠️ Optimized chain setup failed, using basic functionality.")
    return chain