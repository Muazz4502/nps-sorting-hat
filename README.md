# NPS Sorting Hat

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
        |  "Hmm... payments problem... yes, yes...        |
        |   STUCK_TRANSACTIONS_AND_REFUNDS!"              |
         \                                                /
          \                                              /
```

## What it does

You dump in a CSV of NPS responses (or App Store reviews, or Play Store
reviews — the hat doesn't discriminate). It classifies each one into
one of ~25 product categories with high precision.

Crucially: **it returns NULL when it isn't sure.** Most classifiers
give you a confidence score and let you guess the cutoff. This one
admits when it doesn't know. You'd rather have 60% coverage you can
trust than 100% you can't.

## How it works

```
CSV in
  ↓
regex layer        — match obvious patterns first ("refund", "qr code")
  ↓
embedding layer    — sentence-transformers vs. labeled training set
  ↓
LLM layer          — Claude as the tie-breaker for ambiguous cases
  ↓
NULL if any layer disagrees with another
  ↓
CSV out, plus a summary report
```

The three-layer architecture is the trick. Each layer can veto. If
the regex says `STUCK_TRANSACTIONS_AND_REFUNDS` and the embedding says
`PROFILE_AND_ACCOUNT_MANAGEMENT`, both lose. NULL wins. Precision
stays clean.

## Stack

- **Python** · Flask · pandas · sentence-transformers (HuggingFace)
- **Anthropic Claude** for the tie-breaker LLM layer
- **HTML + vanilla JS** front-end (no React, this is a tool not a SaaS)

## Spinning it up

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...      # for the LLM tie-breaker layer
python app.py
```

Open `http://localhost:5000`. Drop a CSV. Watch the hat sort.

## What you get back

- The original CSV with a new `category` column
- A summary: counts per category, top issues, "actionable vs.
  positive vs. NULL"
- An LLM-generated executive summary of the batch ("This week's NPS
  is dominated by stuck transactions on UPI, with a secondary spike in
  KYC re-verification complaints…")

## Customizing for your product

The hat ships with a fintech-app vocabulary. To retune for yours:

1. Edit the **stopwords list** in `nps_classifier.py` — drop your
   product/brand names there so they're treated as noise.
2. Edit the **category map** — the dict that maps user phrases to
   category labels. Add your own product surfaces.
3. Edit the **regex patterns** — the high-precision shortcuts that
   match obvious patterns. Add brand names where helpful.
4. Drop a new training CSV in `training_data/` (or whatever path your
   classifier expects) with columns `feedback,category` to retune the
   embedding layer.

## Why "Sorting Hat"

Because "high-precision multi-label NPS classifier with LLM-based
tie-breaking and confidence-aware NULL fallback" doesn't fit on a
business card.

Also, the response categories *feel* like houses. Some are noble
(`ACTIONABLE_BUG_REPORT`). Some are villainous (`ACCOUNT_BLOCKED`).
Some you don't want to admit you have (`POSITIVE_FEEDBACK` outnumbers
the rest 4:1).

## Status

Internal tool. Used in anger. Not productized. The hat will
occasionally place a Slytherin in Hufflepuff. Set the NULL threshold
high and trust the gaps.
