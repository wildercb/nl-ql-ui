# Prompt Engineering Guide

Welcome to the MPPW-MCP prompt engineering guide. This document explains how to craft effective natural-language (NL) prompts that the platform can reliably translate into GraphQL (QL) queries, as well as how to leverage history, context loading, and multi-agent features to continuously improve results.

---

## Table of Contents

1. Fundamentals
2. Writing Basic NL→QL Prompts
3. Advanced Strategies
4. Using the History Panel
5. Loading External Context
6. Multi-Agent & Enhanced Agents
7. Performance Tips
8. Further Reading

---

## 1. Fundamentals

• **Be explicit.** Mention tables, fields, filters, and ordering directly in your prompt whenever you know them.
• **Keep it one-shot.** A single precise sentence is better than a long paragraph of vague intent.
• **State the desired shape.** Tell the translator which fields or nested objects you want returned.

> Example  
> "List the `logID`, `equipmentID`, and `timestamp` for the 20 most-recent maintenanceLogs ordered by timestamp desc."  
> → Produces a paginated `maintenanceLogs` query limited to 20.

## 2. Writing Basic NL→QL Prompts

1. Identify **entity** (collection / table) you need.  
   *e.g.* `maintenanceLogs`, `thermalScans`, `users`.
2. Specify **fields** to fetch.  
   "include the `notes` and `operatorID` fields".
3. Add **filters/ordering**.  
   "where temperature > 200", "ordered by createdAt desc".
4. Set **pagination**.  
   "first 10", "offset 30".

### Template

```
<Command> <Entity>(<Pagination>) { <Fields> } <Filters/Sorting>
```

Try this structure in natural language:

> "Get the last 5 `thermalScans` with `scanID`, `temperature`, `timestamp` ordered by `timestamp` desc."  

## 3. Advanced Strategies

### Use Variables
If you plan to re-use values, declare them:  
"Use variable `$equip` for equipmentID then fetch logs where equipmentID = `$equip`."

### Combine Queries
The translator supports multi-root queries—ask for two entities in one prompt:  
"Get the 5 latest `maintenanceLogs` and the 5 latest `thermalScans`."

### Mutations
Prefix with an action verb:  
"Create a `maintenanceLog` with `equipmentID`=`printer_XYZ`, `notes`=`Calibrated`, return `logID`."

## 4. Using the History Panel

1. Click **History** in the navigation bar.  
2. Select any previous NL prompt to reload its GraphQL.  
3. Edit and send it again to refine your query.  
4. Use the **Compare** button to see differences between runs.

Leveraging history helps you iteratively converge on the perfect prompt.

## 5. Loading External Context

Sometimes the translator needs extra knowledge:

1. Click **Enhanced Agents** toggle (Home page).  
2. Upload or paste context (e.g. schema docs, sample responses).  
3. Mention it in your prompt:  
   "Using the uploaded schema, list …"  
The agents will inject the context before translation.

## 6. Multi-Agent & Enhanced Agents

• **Multi-Agent** orchestrates several specialised agents (analysis, optimisation, verification).  
• **Enhanced Agents** add domain-specific heuristics.

Turn them on via the purple buttons in the header.  For complex prompts, this can improve accuracy by 10-20 % but incurs slightly higher latency.

## 7. Performance Tips

| Tip | Impact |
| --- | --- |
| Limit result size (e.g. `first 10`) | Faster response, less data transfer |
| Ask for only needed fields | Reduces parsing overhead |
| Cache repetitive queries via History | Zero-latency retrieval |
| Disable Enhanced Agents for trivial prompts | Saves ~1–2 s |

## 8. Further Reading

• [Complete Testing Guide](../../docs/COMPLETE_TESTING_GUIDE.md)  
• [Agent Configuration Guide](../../docs/AGENT_CONFIGURATION_GUIDE.md)

---

_Last updated: 2025-07-05_ 