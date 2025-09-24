
// Neo4j Knowledge Graph Import Script
// Run these commands in Neo4j Browser or Neo4j Desktop

// 1. Create constraints and indexes for better performance
CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT symptom_id IF NOT EXISTS FOR (s:Symptom) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT treatment_id IF NOT EXISTS FOR (t:Treatment) REQUIRE t.id IS UNIQUE;

// 2. Load Disease nodes
LOAD CSV WITH HEADERS FROM "file:///diseases.csv" AS row
CREATE (d:Disease {
    id: toInteger(row.id),
    name: row.name,
    type: row.type
});

// 3. Load Symptom nodes  
LOAD CSV WITH HEADERS FROM "file:///symptoms.csv" AS row
CREATE (s:Symptom {
    id: toInteger(row.id),
    name: row.name,
    type: row.type
});

// 4. Load Treatment nodes
LOAD CSV WITH HEADERS FROM "file:///treatments.csv" AS row
CREATE (t:Treatment {
    id: toInteger(row.id),
    name: row.name,
    type: row.type
});

// 5. Create Disease-Symptom relationships
LOAD CSV WITH HEADERS FROM "file:///disease_symptom_relationships.csv" AS row
MATCH (d:Disease {id: toInteger(row.from_id)})
MATCH (s:Symptom {id: toInteger(row.to_id)})
CREATE (d)-[:HAS_SYMPTOM]->(s);

// 6. Create Disease-Treatment relationships
LOAD CSV WITH HEADERS FROM "file:///disease_treatment_relationships.csv" AS row
MATCH (d:Disease {id: toInteger(row.from_id)})
MATCH (t:Treatment {id: toInteger(row.to_id)})
CREATE (d)-[:TREATED_BY]->(t);

// 7. Verify the import
MATCH (n) RETURN labels(n) AS NodeType, count(n) AS Count;
MATCH ()-[r]->() RETURN type(r) AS RelationshipType, count(r) AS Count;
