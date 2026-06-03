FROM golang:1.22-alpine AS builder

WORKDIR /src

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -mod=readonly -ldflags="-s -w" -o /bin/api ./cmd/api

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -mod=readonly -ldflags="-s -w" -o /bin/healthcheck ./cmd/healthcheck

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -mod=readonly -ldflags="-s -w" -o /bin/preprocess ./cmd/preprocess

RUN /bin/preprocess \
    -refs /src/resources/references.json.gz \
    -mcc /src/resources/mcc_risk.json \
    -norm /src/resources/normalization.json \
    -out /src/resources/references.bin

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /bin/api /api
COPY --from=builder /bin/healthcheck /healthcheck
COPY --from=builder /src/resources/ /app/resources/

EXPOSE 8080

ENTRYPOINT ["/api"]
