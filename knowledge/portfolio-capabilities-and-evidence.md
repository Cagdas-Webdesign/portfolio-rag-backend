---
schema_version: 1
id: portfolio-capabilities-and-evidence
title: Portfolio Capabilities & Evidence
document_type: project
language: de
topics:
  - portfolio
  - evidence
  - interaction-design
  - workflow-design
  - seo
  - campaign-design
technologies:
  - React
  - Vite
  - Node.js
source: "Çağdaş Uçar"
source_type: authored
version: 2
updated_at: 2026-10-09
visibility: public
trust_level: authoritative
---

# Portfolio Capabilities & Evidence

## Portfolio als technischer Kompetenznachweis

Das Portfolio ist nicht nur eine Präsentationsseite, sondern selbst Teil des technischen Kompetenznachweises von Çağdaş Uçar. Es verbindet interaktive Frontend-Erfahrungen mit einem eigenen Backend und einem produktiv betriebenen RAG-System.

Mehrere Portfolio-Bereiche erklären technische oder geschäftliche Abläufe über interaktive Interfaces statt ausschließlich über statischen Text. Dadurch werden Frontend-Entwicklung, API-Integration, AI/RAG, Automation, SEO und Marketing nicht nur genannt, sondern innerhalb des Portfolios visuell und funktional miteinander verbunden.

Die Nachweise sind dabei unterschiedlicher Art. Tatsächlich technisch umgesetzt sind die Verbindung zwischen React-Frontend und eigenem Backend sowie das RAG-System; das RAG-Backend läuft produktiv auf Google Cloud Run. Die interaktiven Bereiche zu Automation, API-Kommunikation sowie SEO und Performance sind Demonstrationen: Sie visualisieren Abläufe innerhalb des Portfolios und sind keine produktiv betriebenen Automations- oder Kundenprozesse. Die Social-Media-Case-Study VERDE ist ein fiktiver Portfolio-Case.

## Frontend und Interaction Design

Çağdaş entwickelt responsive React-Interfaces, bei denen Informationsstruktur, Interaktion und Animation gemeinsam geplant werden. Desktop- und Mobile-Darstellungen werden dabei nicht nur skaliert, sondern abhängig vom verfügbaren Raum unterschiedlich organisiert.

Interaktive Zustände, Scroll-Verhalten und Motion führen Nutzer durch Inhalte und Abläufe. Die Oberfläche dient damit nicht nur der visuellen Gestaltung, sondern auch der verständlichen Vermittlung komplexerer Inhalte.

Als Entwicklungs- und Build-Umgebung des Portfolios dient Node.js mit einem Vite-basierten Frontend-Workflow.

## API- und Systemintegration

Das Portfolio von Çağdaş zeigt API-Integration sowohl konzeptionell als auch praktisch. In den interaktiven Bereichen werden unter anderem Requests, Responses, Authentifizierung, Validierung, Fehlerbehandlung, Webhooks und Synchronisation als Bestandteile der Kommunikation zwischen Systemen dargestellt.

Ein konkreter technischer Nachweis ist das Portfolio selbst: Das React-Frontend ist über eine API mit dem eigenen RAG-Backend verbunden. Die Benutzeroberfläche sendet Anfragen an das Backend und verarbeitet die strukturierte Antwort, während serverseitige Logik und Secrets außerhalb des Frontends bleiben.

Das eigene RAG-Backend bindet seinerseits externe Dienste an: Mistral mit dem Modell `mistral-embed` für die Embeddings, Cloudflare Vectorize als Vector Store und Cloudflare Workers AI für die Textgenerierung. Es ist mit Python und FastAPI entwickelt und läuft produktiv auf Google Cloud Run.

Dadurch wird API-Verständnis nicht nur über Technologiebegriffe beschrieben, sondern innerhalb eines tatsächlich verbundenen Frontend-Backend-Systems demonstriert.

## Automation und Workflow Design

Das Portfolio von Çağdaş visualisiert sowohl klassische Business-Workflows als auch AI-orientierte Abläufe. Prozesse werden dabei als Abfolge aus Triggern, Validierung, Bedingungen, Routing und Aktionen dargestellt.

Die Darstellung berücksichtigt außerdem Fehlerpfade, Retries, Zustände sowie die Übergabe an einen Menschen, wenn ein Prozess nicht vollständig automatisiert fortgeführt werden soll. Automation erscheint so als kontrollierter Prozess mit Erfolgs-, Fehler- und Recovery-Pfaden und nicht lediglich als Verbindung einzelner Tools.

Die gezeigten Abläufe sind eine Portfolio-Demonstration von Workflow- und Systemdenken und nicht automatisch als produktiv eingesetzte Kundenprozesse zu verstehen.

## SEO und Performance

Der SEO- und Performance-Bereich im Portfolio von Çağdaş betrachtet Sichtbarkeit nicht als einzelne Maßnahme, sondern als Zusammenspiel aus technischer Struktur, Inhalt, Indexierbarkeit und Ladeverhalten.

Gezeigt wird unter anderem der Unterschied zwischen der sichtbaren Darstellung für Besucher und der Struktur, die Suchmaschinen verarbeiten können. Semantische Inhalte, Seitenelemente und technische Lesbarkeit werden dabei mit Performance-Aspekten wie Ladepriorisierung, Bildgrößen, Fonts und später geladenen Drittanbieter-Ressourcen verbunden.

Performance wird damit sowohl als technisches Thema als auch als Bestandteil der Nutzererfahrung betrachtet. Im Portfolio dargestellte Beispielwerte dienen der visuellen Erklärung und sind keine gemessenen Produktionskennzahlen.

## Social Media und Campaign Design

Im Social-Media-Bereich zeigt Çağdaş anhand der fiktiven Portfolio-Case-Study „VERDE – Gärten & Landschaft“, wie eine digitale Kampagne über mehrere Berührungspunkte hinweg geplant werden kann.

Die dargestellte Journey reicht von Aufmerksamkeit und Paid-Social-Creative über den Klick auf eine Landingpage bis zu Lead-Qualifizierung, Conversion, Retargeting und der nächsten Kampagnenrunde. Brand System, visuelle Kommunikation, Landingpage und Customer Journey werden dabei als zusammenhängende Bestandteile betrachtet.

Der Case zeigt damit nicht nur Content-Erstellung, sondern die Verbindung von Social Media, Branding, Webdesign, Nutzerführung und Conversion-Denken. VERDE ist eine fiktive Portfolio-Demonstration und kein realer Kundenauftrag.

## Technische und geschäftliche Anforderungen verbinden

Çağdaş betrachtet digitale Produkte nicht ausschließlich aus Sicht einer einzelnen Disziplin. Im Portfolio werden technische und geschäftliche Anforderungen bewusst miteinander verbunden.

Social Media führt beispielsweise nicht nur zu einem Creative, sondern weiter zu Landingpage, Lead-Qualifizierung und Conversion. Frontend-Komponenten werden mit APIs und Backend-Systemen verbunden. Automation berücksichtigt Bedingungen, Fehlerpfade und menschliche Übergaben. SEO und Performance werden gemeinsam mit UX und technischer Struktur betrachtet. Das eigene RAG-System verbindet Backend, AI, Evaluation und Deployment.

So entsteht ein systemorientierter Ansatz, bei dem Design, Entwicklung, Integration und fachliches Ziel als Teile desselben Produkts betrachtet werden.

## Full-Stack als Gesamtzusammenhang

Die Full-Stack-Positionierung von Çağdaş ergibt sich aus der Verbindung mehrerer tatsächlich im Portfolio sichtbarer Ebenen.

Auf der Frontend-Seite entwickelt er React-basierte, responsive und interaktive Benutzeroberflächen. Über definierte APIs werden diese Oberflächen mit serverseitiger Logik verbunden. Das eigene RAG-Backend erweitert diese Architektur um Retrieval, AI-Generierung, Grounding und Quellenvalidierung und wird unabhängig vom Frontend betrieben und deployed.

Das Portfolio demonstriert damit nicht nur isolierte Kenntnisse in Frontend oder Backend, sondern deren Zusammenspiel innerhalb eines vollständigen Web- und AI-Systems.
