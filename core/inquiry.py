"""core/inquiry.py

Deterministic buy-side inquiry / RFQ detector.

Some inbound messages are not supplier OFFERS at all — they are *demand*: a buyer
asking "do you have any offer for X?", "we are looking for a full load of Y",
"please quote Z". These carry no sell-side price list, so running them through
offer extraction yields either a phantom row that gets dropped at save (tripping
the fail-loud "0 persisted" guard in workers/processor.py) or a bare 0-product
"DONE". Both are wrong: there is nothing to sell in the message, and the LLM call
is wasted.

This module classifies such messages up front, deterministically (no LLM, no
credit), so the intake paths can label them status="request" and skip extraction
entirely. The rule is intentionally conservative — a message is only a request
when it shows a clear BUY intent AND carries NO price/offer signal; anything with
a price is treated as a normal offer and left to the extractor.

Used by:
  * core/email_poll.py        — text-only email branch
  * api/whatsapp.py           — _plan_message text branch
  * workers/processor.py      — backstop at the fail-loud guard
"""
from __future__ import annotations

import re
from typing import Dict, Optional

# ── BUY intent: the message is asking to purchase / requesting a quote ──────────
_BUY = re.compile(r"""(?ix)
    \b(
        looking \s+ (?: to \s+ buy | for )                         |
        (?: we | i | we're | we \s+ are | i'm | i \s+ am ) \s+ looking |
        do \s+ you \s+ (?: have | sell | stock | carry )
            \s+ (?: any \s+ | an \s+ )? (?: offer | stock | availability | price ) |
        (?: any | an ) \s+ offer \s+ (?: for | on | of )           |
        can \s+ you \s+ (?: offer | quote | supply | provide | source ) |
        could \s+ you \s+ (?: offer | quote | supply )            |
        (?: please | kindly ) \s+ (?: quote | offer )             |
        (?: we | i ) \s+ (?: need | require |
            want \s+ to \s+ (?: buy | purchase ) |
            would \s+ like \s+ to \s+ (?: buy | purchase ) )      |
        request \s+ for \s+ quotation                             |
        \b r f q \b                                               |
        \b w t b \b                                               |
        in \s+ search \s+ of                                      |
        searching \s+ for                                         |
        interested \s+ (?: in | to ) \s+ (?: buy | buying | purchas ) |
        we \s+ are \s+ buyers                                     |
        (?: any | looking \s+ for \s+ (?:a\s+)? ) \s* (?: full \s+ )? load \s+ (?: for | of ) |
        any \s+ (?: stock | availability ) \s+ (?: of | for )
    )
""", re.VERBOSE)

# ── Price / offer signal: if present the message is almost certainly an OFFER ───
# A currency symbol or ISO code next to a number, an "@ 12.50", or an explicit
# "price ... 12". Pack specs like "6x70cl" or "3 yo" are NOT prices.
_PRICE = re.compile(r"""(?ix)
    [€$£]\s?\d                                               |
    \d\s?[€$£]                                               |
    \b(?:eur|usd|gbp|chf|aud|cad|hkd|sgd|jpy|dkk|sek|nok|pln|czk)\b \s? [:=]? \s? \d |
    \d(?:[.,]\d+)?\s?(?:eur|usd|gbp|chf|€|\$|£)\b            |
    \bprice[sd]?\b[^\n]{0,15}\d                              |
    @\s?\d
""", re.VERBOSE)

# Explicit sell-side phrasing — overrides even a question ("do you need X? we offer it")
_OFFER_PHRASE = re.compile(r"""(?ix)
    \b(
        we \s+ (?: offer | are \s+ offering | can \s+ offer | have \s+ (?: the \s+ following | available | in \s+ stock ) ) |
        our \s+ (?: offer | stock | price \s+ list )  |
        stock \s+ list                               |
        in \s+ stock \s*[:\-]                         |
        available \s+ at                             |
        best \s+ price \s+ (?: is | : )
    )\b
""", re.VERBOSE)

# Pull out what the buyer wants, for a human-readable dashboard label.
_WANTED = re.compile(r"""(?ix)
    (?: looking \s+ for | in \s+ search \s+ of | searching \s+ for |
        we \s+ need | we \s+ require | (?:any\s+)?offer \s+ for | load \s+ for )
    \s+ (.+)
""", re.VERBOSE)


def _wanted(text: str) -> str:
    m = _WANTED.search(text)
    if not m:
        return ""
    frag = re.split(r"[?\n]", m.group(1))[0]
    frag = re.sub(r"\s+", " ", frag).strip(" .,-–")
    return frag[:90]


def classify_inquiry(text: Optional[str]) -> Optional[Dict[str, str]]:
    """Return {"reason": "..."} when `text` is a buy-side inquiry / RFQ with no
    offer content, else None. Conservative: requires BUY intent and NO price or
    sell-side signal, so a genuine offer is never misread as a request."""
    t = (text or "").strip()
    if len(t) < 12:
        return None
    if not _BUY.search(t):
        return None
    if _PRICE.search(t):
        return None
    if _OFFER_PHRASE.search(t):
        return None
    w = _wanted(t)
    reason = f"Buy inquiry (RFQ) — wants: {w}" if w else "Buy inquiry (RFQ) — no sell-side offer"
    return {"reason": reason, "wanted": w}


def is_buy_inquiry(text: Optional[str]) -> bool:
    return classify_inquiry(text) is not None
