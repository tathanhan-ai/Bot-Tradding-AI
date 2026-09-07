# Bai hoc duc ket tu 60 lenh that (paper, BTCUSDT)

So lieu: 60 lenh | SUM +$39.34 | Phi $63.01 | Winrate 38.3% | PF 1.43 | Expectancy +$0.66 | Phi/lai-gop 48.2%.
3 lenh dau (+$91) ganh ca he thong; bo ra la am -$52. 20 lenh gan nhat -$7.44 (WR 35%).

## Bai hoc 1: Dung trung binh gia xuong trong trend nguoc (mat -$55.77)
- Chung cu: lenh 14-18 LONG lien tiep khi gia roi $79,301 -> $78,804: -$12.75, -$4.69, -$5.78, -$7.49, -$25.06.
  Entry context cac lenh thua gan day: Hurst 0.22-0.34 (mean-reverting manh), SMC BEARISH_TREND.
- Nguyen nhan goc: khong co cua cam mo them cung chieu khi dang lo + SMC nguoc huong van cho LONG.
- Bien phap da nap: guard martingale (Stage 4, co san) + `SMCTrendAlignment` moi (Stage 2, `strategy/tactical_lessons.py`):
  SMC BEARISH_TREND cam LONG, BULLISH_TREND cam SHORT.

## Bai hoc 2: FOMO dinh RSI 66 (mat -$7.16 mot lenh)
- Chung cu: lenh 356659 LONG @ $80,155 (RSI 66, sat khang cu) -> -$7.16, lo don le lon nhat nhom CARVER.
- Bien phap da co: FOMO guard RSI 62/38 + can 0.35 ATR (`risk/ai_order_researcher.py`), chi cho POST_ONLY
  chiet khau sau hoac SIDEWAY. Giu nguyen, khong can them.

## Bai hoc 3: Phi an het lai lenh nho (fee/lai-gop 48.2%)
- Chung cu: lenh 1077674 lai gop +$0.11, phi $0.11 -> net -$0.0008. Lenh 1372167 lai +$0.113, phi $0.10 -> +$0.013.
  29/60 lenh chet boi CARVER_REBALANCE (cat non lai mong, moi lan tra phi).
- Bien phap da nap: `FeeAwareEntry` moi (Stage 2): chan lenh co ti le phi/lai > 35%, giam 50% size khi > 20%.
  Cong buffer Carver 25% + notional $350 (co san) giu nguyen.

## Bai hoc 4: Danh directional trong thi truong mean-reverting (14/14 lenh thua)
- Chung cu: toan bo 14 lenh thua gan day co Hurst 0.22-0.34 nhung bot van MARKET/TRAILING duoi trend.
- Bien phap da nap: `HurstRegimeFilter` moi (Stage 2): Hurst < 0.45 cam MARKET/TRAILING/TWAP,
  chi cho Maker/DCA/Grid bat dao chieu; Hurst 0.45-0.55 size tham do.

## Bai hoc 5: Chuoi thua dai khong co thang bao ve (9 thua -$7.09, 8 thua -$11.01)
- Bien phap da co: loss-streak ladder 3/5/7 (cooldown 30p + probation + halt 24h, `ui/server.py`).
  Journal van ghi 25 revenge trades (skill trade-journal, sai so timestamp am) - can giam toc do vao lenh
  sau thua bang cooldown hien tai. Giu nguyen.

## Gate paper de xuat (khong doi)
- Winrate 20 lenh > 45% (hien 35% - CHUA), PF > 1.5 (hien 1.44 - CHUA),
  expectancy > 0 (hien +0.68 - DAT), fee/lai-gop < 30% (hien 48.2% - CHUA).
