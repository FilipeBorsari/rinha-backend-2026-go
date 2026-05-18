package vectorize

import (
	"time"

	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/vectorstore"
)

type Request struct {
	ID          string      `json:"id"`
	Transaction Transaction `json:"transaction"`
	Customer    Customer    `json:"customer"`
	Merchant    Merchant    `json:"merchant"`
	Terminal    Terminal    `json:"terminal"`
	LastTx      *LastTx     `json:"last_transaction"`
}

type Transaction struct {
	Amount       float64 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float64  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float64 `json:"km_from_home"`
}

type LastTx struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float64 `json:"km_from_current"`
}

func Vectorize(r *Request, norm vectorstore.NormConstants, mccRisk map[string]float32) [vectorstore.Dims]float32 {
	var v [vectorstore.Dims]float32

	v[0] = clamp(float32(r.Transaction.Amount) / norm.MaxAmount)

	v[1] = clamp(float32(r.Transaction.Installments) / norm.MaxInstallments)

	var amountVsAvg float32
	if r.Customer.AvgAmount > 0 {
		amountVsAvg = float32(r.Transaction.Amount/r.Customer.AvgAmount) / norm.AmountVsAvgRatio
	}
	v[2] = clamp(amountVsAvg)

	t := parseTime(r.Transaction.RequestedAt)
	v[3] = float32(t.UTC().Hour()) / 23.0

	wd := (int(t.UTC().Weekday()) + 6) % 7
	v[4] = float32(wd) / 6.0

	if r.LastTx == nil {
		v[5] = -1.0
		v[6] = -1.0
	} else {
		lastTime := parseTime(r.LastTx.Timestamp)
		minutes := t.Sub(lastTime).Minutes()
		v[5] = clamp(float32(minutes) / norm.MaxMinutes)
		v[6] = clamp(float32(r.LastTx.KmFromCurrent) / norm.MaxKm)
	}
	v[7] = clamp(float32(r.Terminal.KmFromHome) / norm.MaxKm)
	v[8] = clamp(float32(r.Customer.TxCount24h) / norm.MaxTxCount24h)

	if r.Terminal.IsOnline {
		v[9] = 1.0
	}

	if r.Terminal.CardPresent {
		v[10] = 1.0
	}

	if !merchantKnown(r.Merchant.ID, r.Customer.KnownMerchants) {
		v[11] = 1.0
	}

	if risk, ok := mccRisk[r.Merchant.MCC]; ok {
		v[12] = risk
	} else {
		v[12] = 0.5
	}

	v[13] = clamp(float32(r.Merchant.AvgAmount) / norm.MaxMerchantAvg)

	return v
}

func clamp(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func parseTime(s string) time.Time {
	if len(s) < 19 {
		return time.Time{}
	}
	return time.Date(
		atoi2(s, 0)*100+atoi2(s, 2),
		time.Month(atoi2(s, 5)),
		atoi2(s, 8),
		atoi2(s, 11),
		atoi2(s, 14),
		atoi2(s, 17),
		0,
		time.UTC,
	)
}

func atoi2(s string, i int) int {
	return int(s[i]-'0')*10 + int(s[i+1]-'0')
}

func merchantKnown(merchantID string, known []string) bool {
	for _, k := range known {
		if k == merchantID {
			return true
		}
	}
	return false
}