---
schema_version: 1
id: custom-rag-backend
title: Custom RAG Backend
document_type: project
language: de
topics:
  - rag
  - ai
  - backend
  - retrieval
  - grounding
  - evaluation
technologies:
  - Python
  - FastAPI
  - Mistral
  - Cloudflare Vectorize
  - Cloudflare Workers AI
  - Docker
  - Google Cloud Run
  - REST
source: "Çağdaş Uçar"
source_type: authored
version: 4
updated_at: 2026-09-14
visibility: public
trust_level: authoritative
---

# Custom RAG Backend

Çağdaş Uçars AI Portfolio Assistant wird durch ein eigenes Custom RAG Backend betrieben.

Der Assistant ist nicht als einfacher Chatbot aufgebaut, der lediglich einen großen Master Prompt zusammen mit einer Nutzerfrage an ein Sprachmodell sendet.

Stattdessen besitzt das System eine eigenständige Backend-Architektur für Wissensverarbeitung, Retrieval, Grounding, Quellenkontrolle, Provider-Kommunikation, Evaluation, Security und Testing.

Das Backend wurde mit Python und FastAPI entwickelt.

## Grundprinzip

RAG steht für Retrieval-Augmented Generation.

Das Sprachmodell soll nicht einfach aus seinem allgemeinen Modellwissen heraus Aussagen über Çağdaş erzeugen.

Stattdessen wird relevantes, autorisiertes Wissen aus einer eigenen Knowledge Base gesucht und als kontrollierter Kontext für die Antwort bereitgestellt.

Vereinfacht sieht der End-to-End-Ablauf des AI Portfolio Assistant so aus, vom Eingang der Nutzerfrage bis zur fertigen Antwort mit geprüften Quellenangaben:

Nutzerfrage  
→ API Request  
→ Validierung  
→ Query Embedding  
→ Vector Retrieval  
→ relevante Knowledge Chunks  
→ Context Building  
→ LLM Generation  
→ Grounding  
→ Citation Validation  
→ Backend Response

Damit werden Retrieval und Antwortgenerierung voneinander getrennt.

Diese Abfolge beschreibt den vollständigen Weg einer Frage durch das Backend und damit, wie das System insgesamt funktioniert.

## Knowledge Base

Die Informationen über Çağdaş, seine Fähigkeiten, Projekte und technischen Schwerpunkte liegen nicht mehr als ein einzelner riesiger Master Prompt vor.

Stattdessen besitzt das Backend eine strukturierte Knowledge Base aus getrennten Wissensdokumenten.

Die Dokumente besitzen klar definierte Themen und können unabhängig gepflegt, validiert und indexiert werden.

Dadurch wird Wissen von den eigentlichen Systemanweisungen getrennt.

Fakten gehören in die Knowledge Base.

Verhaltensregeln des Assistants gehören in die dafür vorgesehene Prompt- beziehungsweise Policy-Schicht.

## Knowledge Ingestion

Bevor ein Dokument für Retrieval verwendet werden kann, durchläuft es eine kontrollierte Ingestion-Pipeline.

Dazu gehören unter anderem:

Dokument  
→ Parsing  
→ Metadata  
→ Validation  
→ Chunking  
→ Embedding  
→ Vector Index

Damit wird verhindert, dass beliebiger Inhalt unkontrolliert in den Retrieval-Bestand gelangt.

## Chunking

Knowledge-Dokumente werden für Retrieval in kleinere Einheiten zerlegt.

Diese Einheiten werden als Chunks verarbeitet.

Der Sinn des Chunkings besteht darin, bei einer Nutzerfrage nicht die gesamte Knowledge Base an das Modell zu schicken.

Stattdessen sollen nur die Wissensabschnitte gefunden werden, die für die konkrete Frage relevant sind.

## Embeddings

Die Chunks werden in numerische Vektorrepräsentationen überführt.

Dasselbe Prinzip wird bei einer Nutzerfrage verwendet.

Dadurch kann das System semantische Ähnlichkeit zwischen einer Frage und den vorhandenen Knowledge Chunks bestimmen.

Für den produktiven Embedding-Pfad kommt Mistral mit dem Modell `mistral-embed` zum Einsatz.

## Vector Retrieval

Die Embeddings werden in einem Vector Store gespeichert.

Das Backend verwendet dafür Cloudflare Vectorize.

Bei einer Anfrage wird das Query Embedding mit den gespeicherten Vektoren verglichen und das System sucht nach den relevantesten Knowledge Chunks.

Retrieval entscheidet damit, welches Wissen überhaupt als möglicher Kontext für die Antwort berücksichtigt wird.

## Similarity Threshold

Das System besitzt einen konfigurierbaren Similarity Threshold.

Dieser Threshold wird jedoch bewusst nicht als alleinige Sicherheits- oder Grounding-Grenze betrachtet.

Die Evaluation des Systems hat gezeigt, dass beantwortbare und unbeantwortbare Fragen nicht zuverlässig durch einen einzigen Similarity-Wert getrennt werden können.

Ein Retrieval-Treffer bedeutet deshalb nicht automatisch, dass eine Frage beantwortet werden darf.

Der Threshold ist ein Retrieval-Parameter.

Grounding ist eine separate Schutzschicht.

## Context Building

Aus den gefundenen Chunks wird ein begrenzter Kontext für das Sprachmodell aufgebaut.

Das System besitzt dafür definierte Budgets und Grenzen.

Nicht beliebig viele Dokumente oder unbegrenzt große Textmengen werden an den Provider weitergegeben.

Damit bleibt der Request kontrollierbar und die relevante Information möglichst fokussiert.

## LLM Generation

Nach dem Retrieval erhält das Sprachmodell die kontrollierten Kontextinformationen.

Die Textgenerierung übernimmt Cloudflare Workers AI mit dem Modell `@cf/openai/gpt-oss-120b`.

Das Modell generiert daraus eine mögliche Antwort.

Eine generierte Antwort ist jedoch noch nicht automatisch eine veröffentlichbare Antwort.

Danach folgt die Grounding-Prüfung.

## Grounding

Grounding ist eine zentrale Sicherheits- und Qualitätsgrenze des Systems.

Das Backend prüft, ob eine generierte Antwort tatsächlich durch die bereitgestellten Quellen gestützt werden kann.

Wenn die Antwort nicht ausreichend durch den autorisierten Kontext belegt werden kann, wird sie nicht einfach veröffentlicht.

Dadurch kann das System eine Antwort ablehnen, obwohl Retrieval zuvor Chunks gefunden hat.

Das ist bewusst so gestaltet.

## Unknown Questions

Wenn das System keine ausreichend belegbare Antwort besitzt, soll es nicht improvisieren.

Eine unbekannte Frage ist ein gültiger Systemzustand.

Das Backend kann deshalb eine kontrollierte Antwort ohne Citations zurückgeben, anstatt Informationen über Çağdaş zu erfinden.

Das Ziel lautet nicht, jede Frage zu beantworten.

Das Ziel lautet, nur Antworten zu veröffentlichen, die durch autorisiertes Wissen gestützt werden können.

## Public und Internal Knowledge

Knowledge kann unterschiedliche Sichtbarkeiten besitzen.

Öffentliche Informationen dürfen für Antworten verwendet werden.

Interne Informationen dürfen nicht automatisch Bestandteil einer öffentlichen Antwort werden.

Die Visibility eines Dokuments ist deshalb Teil der Knowledge- und Retrieval-Policy.

## Citation Security

Quellenangaben werden nicht einfach deshalb akzeptiert, weil das Sprachmodell sie erzeugt.

Das Backend kontrolliert Citations selbst.

Ein Modell kann nicht eigenständig eine beliebige Dokument-ID, Section oder URL erfinden und diese dadurch zu einer gültigen Quelle machen.

Nur Quellen, die vom Backend als zulässig und tatsächlich vorhanden erkannt werden, dürfen veröffentlicht werden.

Damit sind Citations backend-owned und nicht model-owned.

## Prompt Injection

Knowledge-Dokumente werden als nicht vertrauenswürdiger Inhalt behandelt.

Ein Dokument darf nicht dadurch Systemrechte erhalten, dass darin beispielsweise steht, vorherige Regeln zu ignorieren.

Dokumentinhalt kann die Rolle des Systems nicht verändern, keine Secrets freigeben und keine Backend-Konfiguration überschreiben.

Gleichzeitig wird Prompt Injection nicht als vollständig gelöst dargestellt.

Ein manipulativer Text kann möglicherweise beeinflussen, was ein Sprachmodell zu generieren versucht.

Die entscheidende Architekturfrage lautet deshalb, was das Modell tatsächlich im System bewirken darf.

Die nachgelagerten Grounding- und Citation-Grenzen kontrollieren, was letztlich veröffentlicht werden darf.

## Provider Architecture

Das Backend ist nicht so aufgebaut, dass die gesamte Kernlogik direkt von einem einzelnen externen Anbieter abhängt.

Provider-spezifische Implementierungen werden über definierte Schnittstellen von der Kernlogik getrennt.

Dadurch können Infrastrukturkomponenten ausgetauscht oder getestet werden, ohne die gesamte Anwendung neu strukturieren zu müssen.

Das System verwendet dafür unter anderem Prinzipien aus Ports & Adapters.

## Composition Root

Die konkrete Zusammensetzung der benötigten Komponenten erfolgt kontrolliert an einer definierten Stelle.

Dadurch wird nicht überall innerhalb der Anwendung spontan entschieden, welcher Provider oder welche Infrastrukturimplementierung verwendet wird.

Diese Trennung verbessert Nachvollziehbarkeit, Testing und Austauschbarkeit.

## FastAPI API

FastAPI bildet die HTTP-Schnittstelle des Backends.

Das React-Portfolio kommuniziert über einen definierten API Contract mit diesem Backend.

Das Frontend muss nicht wissen, wie Retrieval, Embeddings, Vectorize oder Mistral intern funktionieren.

Es muss lediglich den vereinbarten Contract einhalten.

Das ist ein konkretes Beispiel für die Trennung von Frontend und Backend.

## Failure Handling

Das Backend unterscheidet unterschiedliche Fehlerklassen und behandelt Provider- und Netzwerkfehler kontrolliert.

Retry-fähige Fehler werden begrenzt erneut versucht.

Nicht jeder Fehler wird blind wiederholt.

Beispielsweise werden Authentifizierungs- und Berechtigungsfehler nicht durch sinnlose Retries versteckt.

Eine teilweise erzeugte oder technisch unsichere Antwort soll nicht als erfolgreiche Antwort an das Frontend weitergegeben werden.

## Cost Awareness

Externe AI- und Embedding-Aufrufe verursachen Ressourcenverbrauch und potenziell Kosten.

Das Backend versucht deshalb unnötige Provider-Aufrufe zu vermeiden.

Bereits indexierter unveränderter Content muss nicht erneut eingebettet werden.

Ein Dry Run verursacht keine echten Provider-Aufrufe.

Wenn Retrieval bereits feststellt, dass keine ausreichende Wissensbasis vorhanden ist, muss nicht unnötig eine LLM-Generation gestartet werden.

## Evaluation

Das Backend besitzt eine eigene Evaluation Harness.

Retrieval-Qualität und Grounding werden damit nicht nur subjektiv beurteilt.

Ein definiertes Dataset kann verwendet werden, um unter anderem zu untersuchen, ob relevante Dokumente gefunden werden, wie sich Retrieval-Parameter verhalten, welche Fragen scheitern, ob unbekannte Fragen korrekt abgelehnt werden, ob interne Informationen geschützt bleiben und ob Citations korrekt behandelt werden.

Eine wichtige Erkenntnis aus der Evaluation war, dass ein Similarity Threshold allein beantwortbare und unbeantwortbare Fragen nicht zuverlässig trennt.

Deshalb wurde kein scheinbar optimaler Threshold blind aus einem Test-Double als Production-Wert übernommen.

## Testing

Das Backend verfügt über eine umfangreiche automatisierte Test-Suite.

Nach Abschluss der Engineering-Phase bestanden 1296 Tests, während 4 bewusst gegatete Live-Tests übersprungen wurden.

Getestet werden unter anderem Retrieval, Grounding, Prompt-Injection-Szenarien, Citation Security, Fehlerfälle, Retry-Verhalten, API-Verhalten, Konfiguration, Security Boundaries, Container-Verhalten und CLI-Funktionen.

Die Anzahl der Tests ist dabei nicht das eigentliche Qualitätsmerkmal.

Entscheidend ist, dass unterschiedliche Systemgrenzen und Fehlerfälle reproduzierbar überprüft werden können.

## Docker und Production Readiness

Das Backend besitzt einen Docker-basierten Deployment-Pfad.

Die Produktionskonfiguration arbeitet fail-closed: Entwicklungsadapter dürfen nicht unbemerkt als Production-Infrastruktur verwendet werden.

Secrets gehören nicht in das Frontend oder Repository.

CORS wird kontrolliert konfiguriert.

Rate Limiting ist bewusst als Edge-Verantwortung vorgesehen und nicht als scheinbar sicherer In-Process-Limiter hinter unbekannten Proxy-Konfigurationen implementiert.

## Bewusste Grenzen

Das Projekt versucht nicht, künstlich wie eine riesige Enterprise-Plattform auszusehen.

Nicht benötigte Technologien und Funktionen werden nicht nur deshalb eingebaut, damit die Architektur größer wirkt.

Bewusst nicht Bestandteil des aktuellen Systems sind beispielsweise zusätzliche Komponenten wie Conversation Memory, Reranking, Hybrid Search, Redis, Queues, Kubernetes oder eine Admin-Upload-Plattform, solange der konkrete Anwendungsfall sie nicht benötigt.

Das entspricht einem wichtigen Architekturprinzip von Çağdaş:

Komplexität wird nicht mit Qualität verwechselt.

## Portfolio-Relevanz

Der AI Portfolio Assistant demonstriert zwei Ebenen gleichzeitig.

Für einen Besucher ist er eine interaktive Funktion innerhalb des Portfolios.

Technisch demonstriert er dagegen Backend Engineering, API Design, RAG, Embeddings, Vector Retrieval, Provider-Abstraktion, Grounding, Citation Security, Evaluation, Testing und Production Readiness.

Der Assistent ist damit nicht nur ein Element der Portfolio-Oberfläche. Das Backend selbst ist eines der technischen Portfolio-Projekte.

## Technologie-Stack

Der AI Portfolio Assistant beantwortet Fragen von Besuchern auf der Portfolio-Website. Hinter diesem Chatbot steht kein fertiger Chatbot-Dienst, sondern ein eigenes Custom RAG Backend mit einem klar benennbaren Technologie-Stack.

Das Backend ist in Python 3.13 geschrieben. FastAPI bildet die HTTP-Schnittstelle, über die das React-Portfolio den Assistant anspricht.

Für die Embeddings kommt Mistral mit dem Modell `mistral-embed` zum Einsatz. Die Embeddings der Knowledge Chunks liegen in Cloudflare Vectorize, das als Vector Store für das semantische Retrieval dient.

Die Textgenerierung übernimmt Cloudflare Workers AI mit dem Modell `@cf/openai/gpt-oss-120b`. Embeddings und Generation sind damit zwei getrennt konfigurierte Provider.

Ausgeliefert wird das Backend als Docker-Image. Produktiv läuft es als Container auf Google Cloud Run.

Inhaltlich besteht der Assistant aus Knowledge Ingestion, Chunking, Embeddings, Vector Retrieval, Context Building, LLM Generation, Grounding und Citation Validation.

Retrieval, Grounding und Quellenprüfung liegen dabei vollständig im Backend. Das Sprachmodell erhält nur den kontrollierten Kontext und entscheidet weder über die Auswahl der Quellen noch darüber, welche Citations veröffentlicht werden.

## End-to-End-Systemablauf

Diese Section beschreibt den vollständigen End-to-End-Ablauf des AI Portfolio Assistant, vom Eingang einer Nutzerfrage bis zur fertigen, geprüften Antwort.

Vereinfacht läuft eine eingehende Frage durch folgende Schritte:

Nutzerfrage  
→ API Request  
→ Validierung  
→ Query Embedding  
→ Vector Retrieval  
→ relevante Knowledge Chunks  
→ Context Building  
→ LLM Generation  
→ Grounding  
→ Citation Validation  
→ Backend Response

Damit beschreibt dieser Ablauf, wie das System insgesamt funktioniert und welchen Weg eine Frage durch das Backend nimmt, bevor eine geprüfte Antwort an den Nutzer zurückgegeben wird.
