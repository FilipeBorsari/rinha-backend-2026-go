# Rinha de Backend 2026 — Go

## O que é a Rinha?

A **Rinha de Backend** é uma competição onde participantes implementam uma API HTTP com um desafio técnico específico, sob restrições rígidas de recursos (CPU e memória). O objetivo é extrair o máximo de performance dentro dos limites impostos.

Na edição de 2026, o desafio é uma **API de detecção de fraude em tempo real**: dado o payload de uma transação financeira, a API deve retornar um `fraud_score` (0–1) e uma decisão `approved` (true/false) com baixa latência.

O score é calculado via **busca KNN** (k=5, distância euclidiana) sobre um dataset de ~3 milhões de transações de referência, previamente carregadas em memória.

---

## Como será implementada

**Linguagem:** Go (stdlib apenas, sem dependências externas)

**Arquitetura:**

```
porta 9999 (host)
     │
     ▼
  nginx (load balancer)     0.10 CPU / 20 MB
   ┌──────┴──────┐
   ▼             ▼
 api-1          api-2       0.45 CPU / 165 MB cada
 :8080          :8080
```

**Total de recursos permitidos:** 1.00 CPU e 350 MB de RAM.

**Fluxo por requisição:**

1. `POST /fraud-score` recebe o payload JSON da transação
2. O payload é vetorizado em 14 dimensões (`vectorize`)
3. É realizada uma busca KNN no dataset em memória (`vectorstore`)
4. O score é calculado como a proporção de vizinhos rotulados como fraude entre os 5 mais próximos (ex.: 3 fraudes em 5 → `fraud_score = 0.6`)
5. A resposta com `fraud_score` e `approved` é retornada em JSON

**Endpoints:**

| Método | Path | Descrição |
|---|---|---|
| `GET` | `/ready` | Health check — retorna `200 OK` quando o dataset está carregado |
| `POST` | `/fraud-score` | Vetoriza a transação, executa KNN e retorna o score |

**Infraestrutura:**

- Build em dois estágios com Docker (binário estático, imagem `alpine`)
- Duas instâncias da API em paralelo com nginx em round-robin
- Dataset comprimido (`references.json.gz`) carregado inteiramente em memória na inicialização
