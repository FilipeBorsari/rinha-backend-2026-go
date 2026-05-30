# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM golang:1.22-alpine AS builder

WORKDIR /src

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -mod=vendor -ldflags="-s -w" -o /bin/api ./cmd/api

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -mod=vendor -ldflags="-s -w" -o /bin/healthcheck ./cmd/healthcheck

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /bin/api /api
COPY --from=builder /bin/healthcheck /healthcheck

COPY resources/ /app/resources/

EXPOSE 8080

ENTRYPOINT ["/api"]
