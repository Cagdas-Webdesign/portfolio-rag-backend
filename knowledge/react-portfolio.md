---
schema_version: 1
id: react-portfolio
title: React Portfolio
document_type: project
language: de
topics:
  - portfolio
  - frontend
  - ui
  - motion
  - api-integration
technologies:
  - React
  - TypeScript
  - JavaScript
  - HTML
  - CSS
  - Framer Motion
  - GSAP
  - Figma
source: "Çağdaş Uçar"
source_type: authored
version: 1
updated_at: 2026-08-18
visibility: public
trust_level: authoritative
---

# React Portfolio

Çağdaş Uçars Portfolio ist nicht nur eine Präsentationsseite für seine Fähigkeiten. Die Website selbst ist ein technisches Frontend-Projekt und demonstriert seine Arbeit mit moderner Webentwicklung, UI Design, Komponentenarchitektur, Animation und API-Integration.

## Technische Basis

Das Portfolio ist als moderne React-Anwendung aufgebaut.

Zum Frontend-Stack gehören insbesondere React, TypeScript, JavaScript, HTML, CSS, Framer Motion und GSAP.

React bildet die komponentenbasierte Grundlage der Anwendung. TypeScript erweitert die JavaScript-Entwicklung um statische Typisierung und unterstützt eine strukturiertere und besser überprüfbare Codebasis.

## Component Architecture

Das Portfolio wird nicht als eine einzige große Oberfläche betrachtet.

Unterschiedliche Bereiche der Website werden in logisch getrennte Komponenten und Sections aufgeteilt.

Dadurch können Verantwortlichkeiten getrennt, Komponenten wiederverwendet und einzelne Bereiche unabhängig weiterentwickelt werden.

Dazu gehören beispielsweise Sections, UI Components, Navigation, Slider und interaktive Bereiche, Buttons und Controls, Cards, Pop-ups, Animation Components und die Portfolio Assistant UI.

Die Komponentenstruktur soll dabei nicht unnötig kompliziert sein. Eine Komponente erhält eine eigene Verantwortung, wenn die Trennung technisch oder strukturell sinnvoll ist.

## TypeScript und Frontend Contracts

TypeScript wird nicht nur als zusätzliche Syntax betrachtet.

Typen helfen dabei, klar zu definieren, welche Daten eine Komponente erwartet und welche Strukturen innerhalb der Anwendung verarbeitet werden.

Besonders relevant wird dies an Systemgrenzen.

Wenn das React-Frontend mit einem Backend kommuniziert, müssen beide Seiten einen gemeinsamen Contract einhalten.

Das Frontend muss wissen, welche Daten gesendet werden, an welchen Endpoint, mit welcher HTTP-Methode, welche Response erwartet wird, welche Felder vorhanden sein können und welche Fehlerzustände auftreten können.

Dadurch verbindet das Portfolio Frontend Development mit API- und Systemverständnis.

## Figma und UI Design

Die visuelle Entwicklung des Portfolios ist ebenfalls Bestandteil des Projekts.

Çağdaş arbeitet mit Figma und verbindet Designentscheidungen mit der späteren technischen Umsetzung.

Dazu gehören unter anderem Layout, visuelle Hierarchie, Typografie, Spacing, Komponenten, responsive Varianten, Navigation, Interaktionen, moderne UI-Flächen sowie Desktop- und Mobile-Darstellung.

Ein Design wird nicht ausschließlich danach bewertet, ob es als statischer Entwurf gut aussieht.

Es muss sich auch sinnvoll als funktionierende Weboberfläche umsetzen lassen.

## Responsive Design

Das Portfolio wird für unterschiedliche Bildschirmgrößen entwickelt.

Desktop und Mobile werden nicht lediglich als dieselbe Oberfläche in unterschiedlichen Breiten betrachtet.

Komponenten, Abstände, Typografie, Navigation, Animationen und Interaktionen müssen abhängig vom verfügbaren Raum sinnvoll funktionieren.

Responsive Design ist deshalb Bestandteil der Komponenten- und UI-Architektur und keine nachträgliche Korrektur.

## Framer Motion

Framer Motion wird innerhalb des React-Frontends für Animationen und UI-Transitions eingesetzt.

Dadurch können Animationen direkt mit React-Komponenten und deren Zuständen verbunden werden.

Framer Motion kann unter anderem für Einblendungen, Übergänge, Component Transitions, interaktive Zustände, Motion-basierte UI und Viewport-basierte Animationen eingesetzt werden.

## GSAP

Für komplexere Animationen und zeitlich kontrollierte Abläufe setzt Çağdaş zusätzlich GSAP ein.

GSAP eignet sich insbesondere für komplexere Animationssequenzen, Timelines, Scroll-basierte Animationen, präzise kontrollierte Bewegungen und koordinierte Animation mehrerer Elemente.

Framer Motion und GSAP werden dabei nicht eingesetzt, um möglichst viele Effekte auf einer Seite zu erzeugen.

Die Animation soll zur jeweiligen Section, Interaktion und visuellen Geschichte passen.

## Motion als Teil der User Experience

Animation ist im Portfolio nicht ausschließlich Dekoration.

Motion kann dabei helfen, Aufmerksamkeit zu führen, Zustandsänderungen verständlich zu machen, Inhalte schrittweise zu präsentieren, räumliche Zusammenhänge zu vermitteln, Übergänge zwischen UI-Zuständen natürlicher wirken zu lassen und technische Abläufe visuell zu erklären.

Die technische Qualität einer Animation umfasst deshalb nicht nur ihre Optik, sondern auch Timing, Performance, Responsiveness und ihre Funktion innerhalb der Benutzeroberfläche.

## API Integration

Das Portfolio kann mit separaten Backend-Systemen über definierte APIs kommunizieren.

Der Portfolio Assistant ist dafür das wichtigste konkrete Beispiel.

Das React-Frontend ist für die Benutzeroberfläche und Interaktion verantwortlich.

Das RAG-Backend besitzt seine eigenen Verantwortlichkeiten.

Vereinfacht:

Nutzer  
→ React Chat UI  
→ HTTP Request  
→ definierter Backend Endpoint  
→ RAG Backend  
→ Response  
→ React State  
→ Darstellung der Antwort

Frontend und Backend bleiben dadurch technisch voneinander getrennt.

## Keine Secrets im Frontend

Sensible Provider-Credentials gehören nicht in den ausgelieferten React-Code.

Das Frontend kommuniziert deshalb mit einer kontrollierten Backend-Schnittstelle, während Provider-Zugangsdaten und andere Secrets auf der Backend- beziehungsweise Infrastruktur-Seite verbleiben.

Das ist besonders bei KI-, API- und Cloud-Integrationen wichtig.

## Portfolio Assistant Integration

Der AI Portfolio Assistant ist als eigenständige Funktion in die React-Oberfläche integriert.

Die Oberfläche übernimmt unter anderem Nutzereingabe, Request-Auslösung, Loading State, Response-Darstellung, Fehlerzustände, Darstellung gültiger Antworten und die Darstellung von Citations, sofern vorhanden.

Die eigentliche Retrieval-, Grounding- und LLM-Logik liegt dagegen nicht im React-Frontend.

Sie gehört zum separaten Custom-RAG-Backend.

## Trennung der Projekte

Damit demonstriert das Portfolio zwei miteinander verbundene, aber technisch getrennte Projekte.

### React Portfolio

Frontend, TypeScript, UI, Responsive Design, Komponenten, Framer Motion, GSAP, Interaktionen und API-Integration.

### Custom RAG Backend

Python, FastAPI, Knowledge Processing, Embeddings, Vector Retrieval, Grounding, Provider-Integration, Citations, Evaluation, Security und Testing.

Beide Systeme kommunizieren über einen definierten API Contract.

## Positionierung

Das Portfolio zeigt Çağdaş' technische Fähigkeiten nicht ausschließlich durch Texte oder Skill-Listen.

Die Website selbst ist Teil des Nachweises.

Die React-/TypeScript-Architektur, Komponentenstruktur, UI-Umsetzung, Framer-Motion- und GSAP-Animationen sowie die Integration des separaten RAG-Backends sind Bestandteil des technischen Portfolio-Projekts.

Das Portfolio ist damit gleichzeitig Präsentationsoberfläche und praktisches Beispiel seiner Frontend- und Integrationsarbeit.
