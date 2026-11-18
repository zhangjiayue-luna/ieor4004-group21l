# Sensitivity analysis tables

All quantities are probability-weighted over the low/typical/high scenarios and in normalized units. 'Not stocked' is demand for products outside the assortment (part of Unmet). 'Loss @ base wts' re-scores each plan with alpha=5, beta=1, h=0.2, tau=0.5. Rows marked * are the base case. Objective changes smaller than the MIP gaps are marked (~).

## Assortment limit K (products per store)

| Value | Objective | Gap | Fill rate | Unmet | Not stocked | Expired | Carryover | Delivered | Loss @ base wts | Assortment changes |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 1,640.4 | 3.80% | 73.0% | 322.9 | 251.8 | 7.66 | 78.2 | 886.0 | 1,640.4 | 15 |
| 20* | 1,229.3 | 2.16% | 79.6% | 243.6 | 133.6 | 3.46 | 32.6 | 958.1 | 1,229.3 | 0 |
| 25 | 1,019.5 | 0.67% | 83.0% | 203.1 | 46.9 | 1.34 | 10.7 | 994.4 | 1,019.5 | 15 |
| 29 | 964.9 | 0.32% | 83.9% | 192.5 | 0.0 | 0.86 | 6.1 | 1003.9 | 964.9 | 27 |

## Store capacity (x base)

| Value | Objective | Gap | Fill rate | Unmet | Not stocked | Expired | Carryover | Delivered | Loss @ base wts | Assortment changes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.85 | 1,697.0 | 0.10% | 71.6% | 339.3 | 172.8 | 0.20 | 2.1 | 856.4 | 1,697.0 | 14 |
| 1* | 1,229.3 | 2.16% | 79.6% | 243.6 | 133.6 | 3.46 | 32.6 | 958.1 | 1,229.3 | 0 |
| 1.2 | 1,076.7 | 3.22% | 82.5% | 209.3 | 133.6 | 8.64 | 83.1 | 1004.4 | 1,076.7 | 0 |

## Regional supply (x base)

| Value | Objective | Gap | Fill rate | Unmet | Not stocked | Expired | Carryover | Delivered | Loss @ base wts | Assortment changes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.85 | 1,680.5 | 0.00% | 71.9% | 336.1 | 166.8 | 0.00 | 0.3 | 859.4 | 1,680.5 | 14 |
| 1* | 1,229.3 | 2.16% | 79.6% | 243.6 | 133.6 | 3.46 | 32.6 | 958.1 | 1,229.3 | 0 |
| 1.2 | 1,229.3 (~) | 1.97% | 79.6% | 243.6 | 133.6 | 3.43 | 32.4 | 958.0 | 1,229.3 | 0 |

## Shortage penalty α

| Value | Objective | Gap | Fill rate | Unmet | Not stocked | Expired | Carryover | Delivered | Loss @ base wts | Assortment changes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 497.9 | 1.73% | 79.6% | 243.9 | 133.6 | 2.36 | 31.8 | 956.7 | 1,229.6 | 0 |
| 5* | 1,229.3 | 2.16% | 79.6% | 243.6 | 133.6 | 3.46 | 32.6 | 958.1 | 1,229.3 | 0 |
| 10 | 2,447.2 | 2.15% | 79.6% | 243.6 | 133.6 | 3.50 | 32.7 | 958.1 | 1,229.3 | 0 |

## Overnight carryover cost h

| Value | Objective | Gap | Fill rate | Unmet | Not stocked | Expired | Carryover | Delivered | Loss @ base wts | Assortment changes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,222.6 | 1.86% | 79.6% | 243.5 | 133.6 | 3.57 | 37.3 | 958.2 | 1,230.1 | 0 |
| 0.2* | 1,229.3 | 2.16% | 79.6% | 243.6 | 133.6 | 3.46 | 32.6 | 958.1 | 1,229.3 | 0 |
| 0.5 | 1,239.0 | 2.07% | 79.6% | 243.6 | 133.6 | 3.09 | 32.6 | 957.6 | 1,229.2 | 0 |

## Day-7 stock cost τ

| Value | Objective | Gap | Fill rate | Unmet | Not stocked | Expired | Carryover | Delivered | Loss @ base wts | Assortment changes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1,228.6 | 2.01% | 79.6% | 243.6 | 133.6 | 3.46 | 32.6 | 958.1 | 1,229.3 | 0 |
| 0.5* | 1,229.3 | 2.16% | 79.6% | 243.6 | 133.6 | 3.46 | 32.6 | 958.1 | 1,229.3 | 0 |
| 1 | 1,230.6 | 2.02% | 79.6% | 243.6 | 133.6 | 3.29 | 32.7 | 957.9 | 1,229.3 | 0 |
