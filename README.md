# 🎩 NPS Sorting Hat

> A magical creature that reads your survey responses and sorts them
> into houses. The houses are product categories. The magic is
> embeddings + LLMs. The hat is real.

```
                       _....----"""----...._
                  ,-'`                      `'-.
               ,-'                              `-.
             ,'                                    `.
           ,'                                        `.
          /                                            \
         /          NPS  SORTING  HAT                   \
        /                                                \
        |  "Hmm... payments problem... yes, yes…           |
        |   STUCK_TRANSACTIONS_AND_REFUNDS!"               |
         \                                                /
          \                                              /
           `.                                          ,'
             `.                                      ,'
               `-.                                ,-'
```

## What it does

You feed it a CSV of NPS responses. (Or App Store reviews. Or Play
Store reviews. The hat doesn't discriminate.)

It classifies each response into one of ~25 product categories with
**high precision**. Crucially:

> **It returns NULL when it isn't sure.**

Most classifiers give you a confidence score and let you guess the
cutoff. This one admits when it doesn't know.

You'd rather have **60% coverage you can trust** than **100% you
can't**.

## How the sorting works

```
                CSV in
                  ↓
        ┌─────────────────────┐
        │   Regex Layer       │   match obvious patterns
        │   "refund", "qr"…   │   ←── high confidence shortcuts
        └─────────────────────┘
                  ↓
        ┌─────────────────────┐
        │   Embedding Layer   │   sentence-transformers
        │   vs labeled set    │   ←── semantic similarity
        └─────────────────────┘
                  ↓
        ┌─────────────────────┐
        │   LLM Layer         │   Claude as tie-breaker
        │   for ambiguous     │   ←── judgment for edge cases
        └─────────────────────┘
                  ↓
        ┌─────────────────────┐
        │   Disagreement?     │
        │   → NULL.           │   ←── precision guard
        └─────────────────────┘
                  ↓
                 CSV out
                + summary report
                + LLM exec summary
```

The three-layer architecture is the trick. **Each layer can veto.**
If the regex says `STUCK_TRANSACTIONS_AND_REFUNDS` and the embedding
says `PROFILE_AND_ACCOUNT_MANAGEMENT`, both lose. NULL wins.
**Precision stays clean.**

## Stack

🐍 `Python` `Flask` `pandas`
🧠 `sentence-transformers` (HuggingFace) for embeddings
🤖 `Anthropic Claude` for tie-breakers
🌐 `HTML + vanilla JS` front-end (this is a tool, not a SaaS)

## Spinning it up

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-…
python app.py
```

Open `http://localhost:5000`. Drop a CSV. Watch the hat sort.

## What you get back

- The original CSV with a new `category` column
- A summary: counts per category, top issues, "actionable vs
  positive vs NULL"
- An LLM-generated executive summary of the batch:

> *"This week's NPS is dominated by stuck transactions on UPI, with
> a secondary spike in KYC re-verification complaints. Positive
> feedback is unusually quiet — likely a side effect of the dark-mode
> rollout, which has 6 standalone complaints. Recommend prioritising
> the UPI stuck-transaction queue this sprint."*

## Tuning for your product

The hat ships with a fintech-app vocabulary. To retune:

1. **Stopwords** in `nps_classifier.py` — drop your product/brand
   names there so they're treated as noise.
2. **Category map** — the dict that maps user phrases to category
   labels. Add your product surfaces.
3. **Regex patterns** — the high-precision shortcuts. Add brand names
   where helpful.
4. **Training CSV** — drop a new `feedback,category` CSV and retune
   the embedding layer.

## The category houses

A taste of what comes out the other side:

| Category | Vibe |
|---|---|
| `STUCK_TRANSACTIONS_AND_REFUNDS` | "my money is gone" |
| `ACCOUNT_BLOCKED` | "WHY am I locked out" |
| `KYC_AND_VERIFICATION` | "for the third time" |
| `LOGIN_AND_AUTHENTICATION` | "OTPs going to wrong number" |
| `QR_CODE_ISSUES` | "scanner won't read" |
| `PROFILE_AND_ACCOUNT_MANAGEMENT` | "where do I change my address" |
| `ACTIONABLE_BUG_REPORT` | "I think I found a bug" |
| `POSITIVE_FEEDBACK` | "I love this app pls don't change" |

…and ~17 more. The category set is data-driven; modify in
`nps_classifier.py`.

## Why "Sorting Hat"

Because *"high-precision multi-label NPS classifier with LLM-based
tie-breaking and confidence-aware NULL fallback"* doesn't fit on a
business card.

Also, the response categories *feel* like houses. Some are noble
(`ACTIONABLE_BUG_REPORT`). Some are villainous (`ACCOUNT_BLOCKED`).
Some you don't want to admit you have (`POSITIVE_FEEDBACK`
outnumbers everything else 4:1).

## Status

Internal tool. Used in anger. Not productized. The hat will
occasionally place a Slytherin in Hufflepuff. Set the NULL
threshold high and trust the gaps.

---

*Built because reading 800 free-text survey responses by hand was
making me lose the plot.*
