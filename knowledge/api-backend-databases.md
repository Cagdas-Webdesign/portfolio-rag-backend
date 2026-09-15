---
schema_version: 1
id: api-backend-databases
title: API, Backend & Databases
document_type: skill
language: de
topics:
  - backend
  - api
  - databases
  - architecture
  - debugging
technologies:
  - Python
  - FastAPI
  - REST
  - HTTP
  - JSON
  - Webhooks
  - PostgreSQL
  - MySQL
  - MongoDB
  - Supabase
  - Cloudflare Workers
  - Cloudflare Vectorize
  - Docker
source: "Çağdaş Uçar"
source_type: authored
version: 1
updated_at: 2026-08-18
visibility: public
trust_level: authoritative
---

# API, Backend & Databases

Çağdaş Uçar beschäftigt sich mit Backend-Entwicklung, API-Integrationen, Datenbanken und der technischen Kommunikation zwischen unterschiedlichen Systemen.

Sein Schwerpunkt liegt dabei nicht nur auf einzelnen Technologien, sondern auf dem Zusammenspiel von Frontend, API, Backend, Datenbank oder externem Service und Response.

## REST APIs und HTTP-Kommunikation

Çağdaş arbeitet mit REST APIs und versteht, dass funktionierende Kommunikation zwischen zwei Systemen nicht allein daran erkennbar ist, dass ein HTTP-Request technisch erfolgreich beantwortet wird.

Ein Statuscode 200 OK bedeutet zunächst nur, dass der angesprochene Endpoint die Anfrage erfolgreich verarbeitet hat. Er beweist nicht automatisch, dass der richtige Endpoint adressiert wurde, die fachlich richtige Operation ausgeführt wurde, die erwarteten Daten verarbeitet wurden oder Client und Backend denselben Contract verwenden.

Deshalb betrachtet Çağdaş API-Kommunikation immer im Zusammenhang mit dem definierten API Contract.

Dazu gehören korrekter Endpoint, korrekte HTTP-Methode, erwartete Request-Struktur, Request Headers, Parameter und Query Parameters, Authentifizierung wenn erforderlich, erwartete Response-Struktur, Status Codes, fachliche Bedeutung der Response, Fehlerfälle und Validierung.

Ein technisch erfolgreicher Request kann also trotzdem fachlich falsch sein, wenn beispielsweise der falsche Endpoint angesprochen wird oder die Response nicht dem erwarteten Contract entspricht.

## API Contracts

Frontend und Backend müssen klar definieren, wie sie miteinander kommunizieren.

Beispielsweise erwartet ein Endpoint wie POST /api/v1/chat eine bestimmte Request-Struktur und liefert eine definierte Response zurück.

Wenn das Frontend stattdessen einen anderen Endpoint, eine falsche HTTP-Methode oder ein nicht passendes JSON-Format verwendet, ist die Integration nicht korrekt, selbst wenn irgendein Teil des Systems weiterhin einen erfolgreichen HTTP-Status liefert.

Çağdaş betrachtet deshalb Endpoint, Methode, Datenstruktur und Response-Schema als zusammengehörigen Contract.

## API Integration

Çağdaş kann Webanwendungen mit externen APIs und Diensten verbinden.

Eine Integration kann beispielsweise so aufgebaut sein:

React Frontend  
→ eigener Backend-Endpoint  
→ externer API-Provider  
→ Backend-Verarbeitung  
→ definierte Response  
→ Frontend

Dabei müssen Secrets oder API-Keys nicht zwangsläufig im Frontend offengelegt werden. Sensible Provider-Kommunikation kann über eine kontrollierte Backend-Schicht erfolgen.

## Webhooks

Çağdaş arbeitet mit Webhooks für ereignisbasierte Kommunikation zwischen Systemen.

Statt einen Dienst permanent nach Änderungen abzufragen, kann ein System bei einem bestimmten Ereignis eine Anfrage an einen definierten Webhook-Endpoint senden.

Das kann beispielsweise für Formulare, CRM-Prozesse, Automationen, Benachrichtigungen oder externe Services eingesetzt werden.

## Python & FastAPI

Python und FastAPI gehören zu Çağdaş' aktuellem Backend-Stack.

Mit FastAPI können strukturierte HTTP APIs und Backend-Anwendungen entwickelt werden.

Dazu gehören unter anderem API-Routing, Endpoints, Request- und Response-Modelle, Datenvalidierung, Backend-Logik, Fehlerbehandlung, Konfiguration, Provider-Anbindungen, Dependency-Strukturen, API-Schemas und Testing.

Sein Custom-RAG-Backend ist ein konkretes Portfolio-Projekt, bei dem Python und FastAPI innerhalb einer größeren modularen Backend-Architektur eingesetzt werden.

## Backend Architecture und klare Grenzen

Eine vernünftige Backend-Architektur sollte aus Çağdaş' Sicht von Anfang an klar definieren, welche Komponente welche Verantwortung besitzt, welche Schichten miteinander kommunizieren dürfen, welche Daten zwischen Schichten übertragen werden, welche Infrastruktur austauschbar bleiben soll, welche Sicherheitsgrenzen gelten und welche Funktionen bewusst außerhalb des Scopes liegen.

Das Ziel ist nicht, möglichst viele Abstraktionen zu erzeugen.

Das Ziel ist eine Architektur, die verständlich, erweiterbar und innerhalb ihrer definierten Grenzen stabil bleibt.

Mit wachsendem System sollte eine neue Funktion nicht dazu führen, dass fachliche Logik, Provider-Code, HTTP-Verarbeitung und Datenzugriff in derselben Komponente miteinander vermischt werden.

Eine saubere Architektur sollte von Beginn an klar definiert, erweiterbar und innerhalb ihrer eigenen Grenzen nachvollziehbar bleiben.

## Skalierbarkeit durch Struktur

Skalierbarkeit bedeutet dabei nicht automatisch Kubernetes, Microservices, Redis, Kafka, zusätzliche Datenbanken oder möglichst viele Cloud-Dienste.

Ein System kann auch dadurch skalierbarer werden, dass Verantwortlichkeiten klar getrennt sind, Komponenten austauschbar bleiben, Schnittstellen definiert sind, Änderungen lokal begrenzt werden können, Tests einzelne Contracts absichern und neue Funktionen bestehende Bereiche nicht unnötig verändern.

Sein Custom-RAG-Backend ist dafür ein konkretes Beispiel.

Provider-spezifische Infrastruktur wird von der eigentlichen RAG-Logik getrennt. Die API-Schicht kennt nicht die gesamte fachliche Verarbeitung, und einzelne Komponenten besitzen klar abgegrenzte Aufgaben.

## Databases

Çağdaş beschäftigt sich mit unterschiedlichen Datenbanksystemen und Datenmodellen.

Dazu gehören PostgreSQL, MySQL, MongoDB und Supabase.

Er kann Datenbanken im Kontext von Webanwendungen, Formularen, Nutzerdaten, Backend-Systemen, Automationen und digitalen Anwendungen einsetzen und einordnen.

Dabei unterscheidet er zwischen relationalen und dokumentenorientierten Ansätzen und wählt die technische Lösung abhängig vom Anwendungsfall.

## Supabase

Supabase kann als Backend-Plattform unter anderem Datenbankfunktionen und weitere Backend-Services für Webanwendungen bereitstellen.

Çağdaş kann Supabase im Zusammenhang mit Webanwendungen und Datenhaltung einsetzen und mit Frontend- oder Backend-Systemen verbinden.

## Datenflüsse

Ein wichtiger Bestandteil seines Backend-Verständnisses sind nachvollziehbare Datenflüsse.

Beispielsweise:

Formular  
→ Frontend-Validierung  
→ API Request  
→ Backend-Validierung  
→ Datenverarbeitung  
→ Datenbank  
→ Response

oder:

Frontend  
→ eigenes Backend  
→ externer Provider  
→ Provider Response  
→ Backend-Verarbeitung  
→ Frontend

Dadurch wird klar definiert, welches System für welchen Verarbeitungsschritt verantwortlich ist.

## Cloudflare

Cloudflare gehört ebenfalls zu den Technologien, die Çağdaş im Kontext von Web- und Backend-Systemen einsetzt.

Dazu gehören in seinen Projekten insbesondere Cloudflare Workers, Edge-basierte Backend-Funktionen, sichere Provider-Anbindungen, Secrets auf der Backend-Seite, API-Proxying und Cloudflare Vectorize im RAG-Kontext.

Cloudflare Vectorize wird in seinem Custom-RAG-Backend als Vector Store für Embeddings und semantisches Retrieval eingesetzt.

## Docker

Çağdaş arbeitet mit Docker im Kontext moderner Entwicklungs- und Backend-Umgebungen.

Container können verwendet werden, um Anwendungen mit definierten Abhängigkeiten reproduzierbar auszuführen und Entwicklungs- und Deployment-Umgebungen konsistenter aufzubauen.

Sein RAG-Backend verfügt beispielsweise über einen containerisierten Deployment-Pfad und wird auch innerhalb eines frischen Container-Environments getestet.

## Security an Systemgrenzen

Bei API- und Backend-Systemen berücksichtigt Çağdaş, dass insbesondere Systemgrenzen kontrolliert werden müssen.

Dazu gehören je nach Projekt Input Validation, Secrets nicht im Frontend, kontrollierte CORS-Konfiguration, Request- und Response-Validierung, Fehlerbehandlung ohne unnötige interne Informationen, Zugriffskontrolle wenn sie erforderlich ist, Rate Limiting beziehungsweise Edge-Schutz und sichere Provider-Kommunikation.

Security wird dabei nicht als einzelne Funktion betrachtet, sondern als Bestandteil der Architektur.

## Fehlerdiagnose über Systemgrenzen

Bei einem Fehler prüft Çağdaş deshalb nicht nur, ob HTTP 200 zurückkommt.

Er prüft beispielsweise, ob überhaupt der richtige Service angesprochen wurde, ob der richtige Endpoint getroffen wurde, ob die HTTP-Methode stimmt, ob der Request dem erwarteten Schema entspricht, ob das Backend fachlich das Richtige verarbeitet hat, ob die Response dem definierten Schema entspricht und ob das Problem im Frontend, Backend, Provider oder in der Kommunikation dazwischen liegt.

Damit zeigt sich, warum ein Request technisch erfolgreich und trotzdem fachlich komplett falsch sein kann und warum definierte Contracts und Architekturgrenzen überhaupt existieren.

## Positionierung

Çağdaş versteht Backend-Entwicklung als mehr als das Schreiben einzelner Endpoints.

Sein Schwerpunkt liegt darauf, Frontend, APIs, Backend-Logik, Datenbanken, externe Provider und Infrastruktur über klar definierte Schnittstellen miteinander zu verbinden.

Sein Custom-RAG-Backend liefert dafür ein konkretes technisches Portfolio-Beispiel.
