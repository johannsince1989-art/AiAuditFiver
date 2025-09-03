# AI Visibility Auditor — Streamlit App
# ------------------------------------------------------------
# A universal website/page auditor that crawls a URL (optionally shallow),
# extracts on‑page/technical data, runs AI + SEO visibility checks, and
# produces a report with targeted suggestions by page type.
#
# How to run locally:
#   1) pip install -U streamlit requests beautifulsoup4 lxml tldextract
#   2) streamlit run ai-visibility-audit-app.py
#
# Notes:
# - This app fetches server‑rendered HTML (no JS rendering). For SPA/JS heavy
#   sites, results may be partial. Consider pre-rendering or a headless browser
#   for advanced use cases.
# - Crawl limits are intentionally modest to make hosting simple.

import re
import io
import json
import time
import math
import queue
import tldextract
import urllib.parse as urlparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import requests
from bs4 import BeautifulSoup
from urllib import robotparser

import streamlit as st

# -----------------------------
# Utilities
# -----------------------------
DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36 AI-Visibility-Auditor/1.0"
)

TIMEOUT = 15


def normalize_url(url: str) -> str:
    if not re.match(r"^https?://", url, flags=re.I):
        return "https://" + url
    return url


def get_base(url: str) -> str:
    parts = urlparse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def same_domain(a: str, b: str) -> bool:
    ea = tldextract.extract(a)
    eb = tldextract.extract(b)
    return (ea.domain, ea.suffix) == (eb.domain, eb.suffix)


# -----------------------------
# Fetching
# -----------------------------
@dataclass
class FetchResult:
    url: str
    status: Optional[int]
    html: str
    headers: Dict[str, str]
    elapsed: float
    error: Optional[str] = None


def fetch(url: str, user_agent: str = DEFAULT_UA) -> FetchResult:
    start = time.time()
    try:
        r = requests.get(url, headers={"User-Agent": user_agent}, timeout=TIMEOUT, allow_redirects=True)
        elapsed = time.time() - start
        html = r.text if r.status_code == 200 else r.text
        return FetchResult(url=r.url, status=r.status_code, html=html, headers=dict(r.headers), elapsed=elapsed)
    except Exception as e:
        return FetchResult(url=url, status=None, html="", headers={}, elapsed=time.time() - start, error=str(e))


# -----------------------------
# robots.txt & sitemap
# -----------------------------
@dataclass
class RobotsInfo:
    robots_url: str
    is_allowed: Optional[bool]
    crawl_delay: Optional[float]
    sitemaps: List[str]
    robots_fetched: bool
    error: Optional[str] = None


def parse_robots(base_url: str, target_url: str, user_agent: str = DEFAULT_UA) -> RobotsInfo:
    robots_url = urlparse.urljoin(base_url, "/robots.txt")
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        # manually fetch to preserve headers & errors
        r = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=TIMEOUT)
        if r.status_code == 200:
            text = r.text
            rp.parse(text.splitlines())
            # extract sitemap lines
            sitemaps = []
            for line in text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemaps.append(line.split(":", 1)[1].strip())
            allowed = rp.can_fetch(user_agent, target_url)
            cd = None
            try:
                cd = rp.crawl_delay(user_agent)
            except Exception:
                cd = None
            return RobotsInfo(robots_url, allowed, cd, sitemaps, True, None)
        else:
            return RobotsInfo(robots_url, None, None, [], False, f"HTTP {r.status_code}")
    except Exception as e:
        return RobotsInfo(robots_url, None, None, [], False, str(e))


@dataclass
class SitemapInfo:
    sitemap_urls: List[str]
    discovered_urls: List[str]
    fetched: bool
    error: Optional[str] = None


def fetch_sitemap_candidates(base_url: str) -> List[str]:
    # typical locations
    return [
        urlparse.urljoin(base_url, "/sitemap.xml"),
        urlparse.urljoin(base_url, "/sitemap_index.xml"),
        urlparse.urljoin(base_url, "/sitemap-index.xml"),
    ]


def _parse_sitemap_xml(xml_text: str) -> List[str]:
    # Simple XML parsing with regex to avoid extra deps. Handles <loc>…</loc> in urlset or sitemapindex.
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml_text, flags=re.I)
    return list(dict.fromkeys(locs))  # de-duplicate, keep order


def get_sitemap_urls(base_url: str, robots_info: RobotsInfo, user_agent: str = DEFAULT_UA) -> SitemapInfo:
    urls: List[str] = []
    errors: List[str] = []
    candidates = list(robots_info.sitemaps) if robots_info and robots_info.sitemaps else []
    candidates += [c for c in fetch_sitemap_candidates(base_url) if c not in candidates]

    discovered: List[str] = []
    fetched_any = False

    for sm_url in candidates:
        try:
            r = requests.get(sm_url, headers={"User-Agent": user_agent}, timeout=TIMEOUT)
            if r.status_code == 200 and ("xml" in r.headers.get("Content-Type", "").lower() or r.text.strip().startswith("<")):
                fetched_any = True
                ls = _parse_sitemap_xml(r.text)
                if any(x.endswith('.xml') for x in ls):
                    # sitemap index: add nested sitemaps
                    for s in ls:
                        if s not in urls:
                            urls.append(s)
                else:
                    # urlset: add page urls
                    for u in ls:
                        if u not in discovered:
                            discovered.append(u)
                if sm_url not in urls:
                    urls.append(sm_url)
            else:
                errors.append(f"{sm_url}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"{sm_url}: {e}")

    return SitemapInfo(sitemap_urls=urls, discovered_urls=discovered, fetched=fetched_any, error="; ".join(errors) if errors else None)


# -----------------------------
# Extraction
# -----------------------------
@dataclass
class PageData:
    url: str
    final_url: str
    status: Optional[int]
    title: str
    meta_description: Optional[str]
    meta_robots: Optional[str]
    canonical: Optional[str]
    lang: Optional[str]
    headings: Dict[str, List[str]]
    body_text: str
    lists: List[List[str]]
    images: List[Dict]
    links_internal: List[Dict]
    links_external: List[Dict]
    og: Dict[str, str]
    twitter: Dict[str, str]
    structured_data: List[Dict]
    fetch_time_ms: int
    headers: Dict[str, str]


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def extract_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def extract_json_ld(soup: BeautifulSoup) -> List[Dict]:
    items = []
    for tag in soup.find_all("script", attrs={"type": re.compile(r"application/(ld\+json|json)" , re.I)}):
        try:
            txt = tag.string or tag.get_text() or ""
            txt = txt.strip()
            if not txt:
                continue
            # Some sites use multiple JSON objects concatenated; try to load safely.
            # Attempt: if it looks like an array, parse as list; else as single object.
            if txt.startswith("["):
                data = json.loads(txt)
                if isinstance(data, list):
                    items.extend(data)
                else:
                    items.append(data)
            else:
                data = json.loads(txt)
                items.append(data)
        except Exception:
            # ignore malformed JSON-LD
            continue
    return items


def extract_meta(soup: BeautifulSoup) -> Tuple[Dict[str, str], Dict[str, str], Optional[str]]:
    og = {}
    tw = {}
    robots = None
    for m in soup.find_all("meta"):
        name = (m.get("name") or m.get("property") or "").lower()
        content = m.get("content")
        if not content:
            continue
        if name.startswith("og:"):
            og[name] = content
        elif name.startswith("twitter:"):
            tw[name] = content
        elif name == "robots":
            robots = content
    return og, tw, robots


def extract_images(soup: BeautifulSoup, page_url: str) -> List[Dict]:
    imgs = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        alt = img.get("alt")
        if not src:
            continue
        abs_url = urlparse.urljoin(page_url, src)
        fname = urlparse.urlsplit(abs_url).path.split("/")[-1]
        imgs.append({"src": abs_url, "alt": alt or "", "filename": fname})
    # figure/figcaption
    for fig in soup.find_all("figure"):
        cap = fig.find("figcaption")
        if cap:
            text = clean_text(cap.get_text())
            im = fig.find("img")
            if im:
                src = im.get("src") or ""
                if src:
                    abs_url = urlparse.urljoin(page_url, src)
                    fname = urlparse.urlsplit(abs_url).path.split("/")[-1]
                    imgs.append({"src": abs_url, "alt": im.get("alt") or "", "caption": text, "filename": fname})
    return imgs


def extract_links(soup: BeautifulSoup, page_url: str) -> Tuple[List[Dict], List[Dict]]:
    base = get_base(page_url)
    internals, externals = [], []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        href = urlparse.urljoin(page_url, href)
        text = clean_text(a.get_text())
        rel = (a.get("rel") or [])
        nofollow = "nofollow" in [r.lower() for r in rel]
        d = {"href": href, "text": text, "nofollow": nofollow}
        if same_domain(href, base):
            internals.append(d)
        else:
            externals.append(d)
    return internals, externals


def extract_headings(soup: BeautifulSoup) -> Dict[str, List[str]]:
    out = {}
    for level in range(1, 7):
        tag = f"h{level}"
        out[tag] = [clean_text(h.get_text()) for h in soup.find_all(tag)]
    return out


def main_text(soup: BeautifulSoup) -> Tuple[str, List[List[str]]]:
    # Simple body text & lists; not using heavy readability deps to keep install light
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    lists = []
    for lst in soup.find_all(["ul", "ol"]):
        items = [clean_text(li.get_text()) for li in lst.find_all("li")]
        if items:
            lists.append(items)
    body = clean_text(soup.get_text(" "))
    return body, lists


def extract_page(url: str, user_agent: str = DEFAULT_UA) -> Tuple[Optional[PageData], Optional[str]]:
    res = fetch(url, user_agent=user_agent)
    if res.error:
        return None, f"Fetch error: {res.error}"

    soup = extract_soup(res.html)
    title = clean_text(soup.title.get_text()) if soup.title else ""

    # meta description
    md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    meta_description = clean_text(md.get("content")) if md and md.get("content") else None

    # lang
    lang = None
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        lang = html_tag.get("lang")

    # canonical
    canonical = None
    can = soup.find("link", rel=re.compile(r"canonical", re.I))
    if can and can.get("href"):
        canonical = urlparse.urljoin(res.url, can.get("href"))

    og, tw, robots = extract_meta(soup)
    headings = extract_headings(soup)
    body_text, lists = main_text(soup)
    images = extract_images(soup, res.url)
    li_int, li_ext = extract_links(soup, res.url)
    structured = extract_json_ld(soup)

    pd = PageData(
        url=url,
        final_url=res.url,
        status=res.status,
        title=title,
        meta_description=meta_description,
        meta_robots=robots,
        canonical=canonical,
        lang=lang,
        headings=headings,
        body_text=body_text,
        lists=lists,
        images=images,
        links_internal=li_int,
        links_external=li_ext,
        og=og,
        twitter=tw,
        structured_data=structured,
        fetch_time_ms=int(res.elapsed * 1000),
        headers=res.headers,
    )
    return pd, None


# -----------------------------
# Heuristics: AI Readability / SEO / Content
# -----------------------------
VOWELS = "aeiouy"


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    # Basic heuristic syllable counter
    count = 0
    prev_vowel = False
    for ch in w:
        if ch in VOWELS:
            if not prev_vowel:
                count += 1
                prev_vowel = True
        else:
            prev_vowel = False
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)
    if not words:
        return 0.0
    sentences = re.split(r"[.!?]+\s+", text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        sentences = [text]
    syllables = sum(count_syllables(w) for w in words)
    W = len(words)
    S = max(len(sentences), 1)
    return 206.835 - 1.015 * (W / S) - 84.6 * (syllables / W)


def est_read_time_minutes(text: str) -> float:
    WPM = 200
    words = len(re.findall(r"\w+", text))
    return round(words / WPM, 2)


def detect_page_type(pd: PageData) -> str:
    types = set()
    for item in pd.structured_data:
        t = item.get("@type") if isinstance(item, dict) else None
        if isinstance(t, list):
            types.update([str(x) for x in t])
        elif t:
            types.add(str(t))
    t_low = {t.lower() for t in types}
    title_l = pd.title.lower()
    body_l = pd.body_text.lower()[:1000]

    if {"product"} & t_low or any(k in title_l for k in ["price", "review", "buy"]) or "add to cart" in body_l:
        return "ecommerce_product"
    if {"article", "blogposting", "newsarticle"} & t_low or any(k in title_l for k in ["blog", "how to", "guide", "news"]):
        return "blog_article"
    if {"service"} & t_low or any(k in title_l for k in ["services", "consulting", "agency", "repair", "plumbing", "law", "attorney", "doctor"]):
        return "service"
    if {"softwareapplication"} & t_low or "pricing" in body_l or "free trial" in body_l:
        return "saas"
    return "generic"


@dataclass
class Score:
    name: str
    value: int
    out_of: int
    notes: List[str]


@dataclass
class Report:
    page_type: str
    ai_readability: Score
    seo: Score
    content: Score
    highlights: List[str]
    issues: List[str]
    suggestions: List[str]
    raw: Dict


# Helper checks

def has_faq(pd: PageData) -> bool:
    for item in pd.structured_data:
        if isinstance(item, dict) and str(item.get("@type", "")).lower() in {"faqpage", "qapage"}:
            return True
    # fallback: headings containing FAQ
    return any("faq" in h.lower() for h in pd.headings.get("h2", []) + pd.headings.get("h3", []))


def keyword_candidates(pd: PageData) -> List[str]:
    # naive: top repeated non-stopwords (short list)
    text = (pd.title + " \n" + (pd.meta_description or "") + " \n" + pd.body_text)
    words = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text)]
    stop = set("""
        the a an and or but with for from this that these those your our their you we us are is was were be have has had to of on in at by as it its they them i me my he she his her not no can will just about more most other into over under after before than within without between up down off across near per vs versus
    """.split())
    freq: Dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    # sort by frequency and length to favor meaningful phrases
    candidates = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))[:15]
    return [w for w, _ in candidates]


def ai_readability_layer(pd: PageData) -> Score:
    notes = []
    score = 0
    out_of = 10

    fre = flesch_reading_ease(pd.body_text)
    read_time = est_read_time_minutes(pd.body_text)
    if fre >= 60:
        score += 3; notes.append(f"Good readability (Flesch ≈ {fre:.0f}).")
    elif fre >= 40:
        score += 2; notes.append(f"Moderate readability (Flesch ≈ {fre:.0f}).")
    else:
        notes.append(f"Dense/technical writing (Flesch ≈ {fre:.0f}). Consider simpler sentences.")

    if pd.headings.get("h1"):
        score += 2; notes.append("H1 present and parseable.")
    else:
        notes.append("Missing H1 or not detectable.")

    if has_faq(pd):
        score += 1; notes.append("FAQ/Q&A structure detected (helps GenAI answers).")

    # entity/intent hints: title+schema presence
    if pd.title and len(pd.title) >= 20:
        score += 1; notes.append("Title expresses intent.")
    if any(isinstance(x, dict) and x.get("@type") for x in pd.structured_data):
        score += 1; notes.append("JSON-LD schema present (structured facts).")

    # info density via lists/tables
    if len(pd.lists) >= 1:
        score += 1; notes.append("Lists detected—good for concise extraction.")

    # brand/entity mentions
    brand_mentioned = any(
        isinstance(x, dict) and (x.get("brand") or x.get("name") or x.get("publisher")) for x in pd.structured_data
    )
    if brand_mentioned:
        score += 1; notes.append("Brand/Entity present in schema.")
    else:
        notes.append("Add clear brand/entity references in copy or schema.")

    return Score("AI Readability", score, out_of, notes)


def seo_layer(pd: PageData, robots: RobotsInfo, sitemap: SitemapInfo) -> Score:
    notes = []
    score = 0
    out_of = 12

    # Title & meta
    if pd.title:
        length = len(pd.title)
        if 30 <= length <= 65:
            score += 2; notes.append(f"Title length good ({length} chars).")
        else:
            notes.append(f"Title length suboptimal ({length} chars). Aim ~50–60.")
    else:
        notes.append("Missing <title>.")

    if pd.meta_description:
        l2 = len(pd.meta_description)
        if 110 <= l2 <= 160:
            score += 2; notes.append(f"Meta description length good ({l2} chars).")
        else:
            notes.append(f"Meta description length {l2} chars. Aim ~120–155.")
    else:
        notes.append("Missing meta description.")

    # Canonical
    if pd.canonical:
        if pd.canonical.split('#')[0].rstrip('/') == pd.final_url.split('#')[0].rstrip('/'):
            score += 1; notes.append("Canonical points to this URL.")
        else:
            notes.append("Canonical points elsewhere—confirm duplication intent.")
    else:
        notes.append("Missing canonical link tag.")

    # Robots meta
    if pd.meta_robots and 'noindex' in pd.meta_robots.lower():
        notes.append("Meta robots NOINDEX present—page may be excluded from search.")
    else:
        score += 1; notes.append("Page indexable via meta robots.")

    # OpenGraph/Twitter
    if pd.og:
        score += 1; notes.append("OpenGraph tags present.")
    if pd.twitter:
        score += 1; notes.append("Twitter Card tags present.")

    # Headings quality
    h1s = pd.headings.get("h1", [])
    if len(h1s) == 1:
        score += 1; notes.append("Single H1 present.")
    elif len(h1s) > 1:
        notes.append("Multiple H1s detected—prefer one primary.")
    else:
        notes.append("No H1 detected.")

    # Images alt coverage
    images = [im for im in pd.images if im.get("src")]
    if images:
        with_alt = sum(1 for im in images if im.get("alt"))
        ratio = with_alt / len(images)
        if ratio >= 0.7:
            score += 1; notes.append(f"Alt text coverage good ({ratio:.0%}).")
        else:
            notes.append(f"Alt text coverage low ({ratio:.0%}).")

    # Links & crawlability
    if robots and robots.is_allowed is not None:
        if robots.is_allowed:
            score += 1; notes.append("Allowed by robots.txt for this UA.")
        else:
            notes.append("Blocked by robots.txt for this UA.")
    else:
        notes.append("robots.txt not found or unreadable.")

    if sitemap and sitemap.fetched:
        score += 1; notes.append("Sitemap discovered.")
    else:
        notes.append("No sitemap found at standard locations or robots.txt.")

    # Internal links presence
    if len(pd.links_internal) >= 5:
        score += 1; notes.append("Healthy internal linking on page.")
    else:
        notes.append("Few internal links—consider adding contextual links.")

    return Score("SEO", score, out_of, notes)


def content_layer(pd: PageData) -> Score:
    notes = []
    score = 0
    out_of = 10

    # Clarity: H2/H3 coverage
    if len(pd.headings.get("h2", [])) >= 2:
        score += 2; notes.append("H2s structure the content well.")
    else:
        notes.append("Add subheadings (H2/H3) to organize topics.")

    # Uniqueness proxy: length & specificity
    if len(pd.body_text) > 1200:
        score += 2; notes.append("Substantial body content present.")
    elif len(pd.body_text) > 300:
        score += 1; notes.append("Some body content present—could expand for completeness.")
    else:
        notes.append("Very short page—expand content for depth.")

    # Authority signals: author/org/reviews
    text_l = pd.body_text.lower()
    authorish = re.search(r"by\s+[A-Z][a-z]+\s+[A-Z][a-z]+", pd.body_text)
    has_org_schema = any(isinstance(x, dict) and str(x.get("@type", "")).lower() in {"organization", "brand"} for x in pd.structured_data)
    has_reviews = any(isinstance(x, dict) and (x.get("aggregateRating") or x.get("review")) for x in pd.structured_data)

    if authorish or "author" in text_l:
        score += 1; notes.append("Author information present.")
    if has_org_schema:
        score += 1; notes.append("Organization/Brand schema present.")
    if has_reviews:
        score += 1; notes.append("Reviews/Ratings detected.")

    # Topical completeness: lists and FAQ
    if len(pd.lists) >= 1:
        score += 1; notes.append("Bulleted/numbered lists add scannability.")
    if has_faq(pd):
        score += 1; notes.append("FAQ provides direct answers.")

    return Score("Content", score, out_of, notes)


# Suggestions by page type

def suggestions_for_type(page_type: str, pd: PageData) -> List[str]:
    s: List[str] = []
    if page_type == "ecommerce_product":
        s += [
            "Add/verify Product schema (name, brand, sku, offers with price & availability, aggregateRating/review).",
            "Ensure high‑quality images with descriptive alt text (color, angle, feature).",
            "Surface key specs in a bullet list or table for LLM extraction.",
            "Include FAQs (shipping, returns, sizing) and clearly marked price.",
        ]
    elif page_type == "service":
        s += [
            "Include clear list of services, benefits, and service areas.",
            "Add LocalBusiness/Service schema with NAP + geo coordinates where relevant.",
            "Add testimonials/case studies and a concise CTA (contact, quote).",
            "Create a FAQ answering common objections.",
        ]
    elif page_type == "blog_article":
        s += [
            "Use Article/BlogPosting schema with author, datePublished, dateModified.",
            "Add a table of contents (H2/H3) and a summary for LLMs.",
            "Add FAQ schema for key questions the post answers.",
            "Cite authoritative sources with outbound links.",
        ]
    elif page_type == "saas":
        s += [
            "Add SoftwareApplication/Product schema including offers/pricing tiers.",
            "Provide comparison tables and feature checklists for easy parsing.",
            "Include demo/trial CTAs and security/compliance facts in bullets.",
            "Add FAQs about pricing, integrations, onboarding.",
        ]
    else:
        s += [
            "Ensure clear H1 and descriptive title/meta.",
            "Use JSON-LD schema that matches the page purpose.",
            "Use lists/tables to pack structured facts for LLMs.",
        ]
    return s


def build_report(pd: PageData, robots: RobotsInfo, sitemap: SitemapInfo) -> Report:
    pt = detect_page_type(pd)
    ai = ai_readability_layer(pd)
    seo = seo_layer(pd, robots, sitemap)
    cont = content_layer(pd)

    highlights: List[str] = []
    issues: List[str] = []

    # Highlights & Issues synthesized from notes
    for n in ai.notes + seo.notes + cont.notes:
        if any(k in n.lower() for k in ["good", "present", "detected", "allowed", "healthy", "points to this"]):
            highlights.append(n)
        if any(k in n.lower() for k in ["missing", "low", "blocked", "no ", "suboptimal", "elsewhere", "few", "dense", "short"]):
            issues.append(n)

    # Suggestions
    suggestions = []
    suggestions += suggestions_for_type(pt, pd)

    # Generic suggestions derived from issues
    for n in issues:
        if "missing <title>" in n.lower():
            suggestions.append("Add an informative, unique <title> (~50–60 chars).")
        if "meta description" in n.lower():
            suggestions.append("Write a compelling meta description (~120–155 chars).")
        if "canonical" in n.lower():
            suggestions.append("Add/verify canonical URL; self‑reference if this is the primary page.")
        if "alt text" in n.lower():
            suggestions.append("Add descriptive alt text to images focusing on product/features.")
        if "internal links" in n.lower():
            suggestions.append("Add relevant internal links to related topics/products.")
        if "robots.txt" in n.lower():
            suggestions.append("Ensure robots.txt is accessible and not unintentionally blocking this path.")

    raw = {
        "page_data": pd.__dict__,
        "robots": robots.__dict__ if robots else None,
        "sitemap": sitemap.__dict__ if sitemap else None,
        "keywords_candidate": keyword_candidates(pd),
        "reading_ease": flesch_reading_ease(pd.body_text),
    }

    return Report(page_type=pt, ai_readability=ai, seo=seo, content=cont, highlights=highlights, issues=issues, suggestions=suggestions, raw=raw)


# -----------------------------
# Crawl (shallow)
# -----------------------------
@st.cache_data(show_spinner=False)
def crawl(start_url: str, max_pages: int, max_depth: int, user_agent: str) -> Dict[str, PageData]:
    start_url = normalize_url(start_url)
    base = get_base(start_url)
    visited: Set[str] = set()
    results: Dict[str, PageData] = {}

    q: queue.Queue[Tuple[str, int]] = queue.Queue()
    q.put((start_url, 0))

    while not q.empty() and len(results) < max_pages:
        url, depth = q.get()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        pd, err = extract_page(url, user_agent=user_agent)
        if pd:
            results[url] = pd
            # enqueue internal links
            if depth < max_depth:
                for l in pd.links_internal:
                    href = l.get("href")
                    if href and same_domain(href, base) and href not in visited:
                        # avoid fragments and mailto
                        if href.startswith("http") and not href.startswith("mailto:"):
                            q.put((href.split("#")[0], depth + 1))
        time.sleep(0.2)  # be polite

    return results


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Visibility Auditor", page_icon="🔎", layout="wide")

st.title("🔎 AI Visibility Auditor (AI + SEO)")
st.caption("Crawl a URL, extract content & technical data, and get an AI‑ready visibility report.")

with st.sidebar:
    st.header("Settings")
    default_url = st.text_input("Start URL", placeholder="https://example.com/page", value="")
    max_pages = st.slider("Max pages to crawl", 1, 25, 3)
    max_depth = st.slider("Max crawl depth", 0, 2, 1, help="0 = single page, 1 = follow internal links once")
    ua = st.text_input("User‑Agent", value=DEFAULT_UA)
    run = st.button("Run Audit", type="primary")

if run:
    if not default_url:
        st.error("Please enter a URL.")
        st.stop()

    with st.spinner("Crawling & analyzing…"):
        pages = crawl(default_url, max_pages=max_pages, max_depth=max_depth, user_agent=ua)

    if not pages:
        st.error("No pages fetched. Check the URL or robots.txt restrictions.")
        st.stop()

    # robots/sitemap only need base
    base_url = get_base(normalize_url(default_url))
    robots = parse_robots(base_url, normalize_url(default_url), user_agent=ua)
    sitemap = get_sitemap_urls(base_url, robots, user_agent=ua)

    # Analyze primary page first
    main_pd = list(pages.values())[0]
    report = build_report(main_pd, robots, sitemap)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Page Type", report.page_type.replace("_", " ").title())
    with c2:
        st.metric("AI Readability", f"{report.ai_readability.value}/{report.ai_readability.out_of}")
    with c3:
        st.metric("SEO", f"{report.seo.value}/{report.seo.out_of}")
    with c4:
        st.metric("Content", f"{report.content.value}/{report.content.out_of}")

    st.subheader("Highlights ✅")
    if report.highlights:
        for h in report.highlights:
            st.write("• ", h)
    else:
        st.write("—")

    st.subheader("Issues ⚠️")
    if report.issues:
        for i in report.issues:
            st.write("• ", i)
    else:
        st.write("—")

    st.subheader("Suggestions 📈")
    if report.suggestions:
        for s in report.suggestions:
            st.write("• ", s)
    else:
        st.write("—")

    with st.expander("🔤 Keywords (candidates)"):
        st.write(", ".join(report.raw["keywords_candidate"]))

    with st.expander("📄 Technical & On‑Page Data (main page)"):
        pd = main_pd
        st.markdown(f"**Final URL:** {pd.final_url}")
        st.markdown(f"**Status:** {pd.status}")
        st.markdown(f"**Lang:** {pd.lang or '—'}  ")
        st.markdown(f"**Title:** {pd.title or '—'}  ")
        st.markdown(f"**Meta Description:** {pd.meta_description or '—'}  ")
        st.markdown(f"**Meta Robots:** {pd.meta_robots or '—'}  ")
        st.markdown(f"**Canonical:** {pd.canonical or '—'}  ")

        st.markdown("**OpenGraph tags:**")
        st.json(pd.og or {})
        st.markdown("**Twitter tags:**")
        st.json(pd.twitter or {})
        st.markdown("**JSON‑LD (structured data):**")
        st.json(pd.structured_data or [])

        st.markdown("**Headings:**")
        st.json(pd.headings)

        st.markdown("**Lists:**")
        st.json(pd.lists)

        st.markdown("**Images (src, alt, filename):**")
        st.dataframe(pd.images)

        st.markdown("**Internal Links (first 100):**")
        st.dataframe(pd.links_internal[:100])

        st.markdown("**External Links (first 100):**")
        st.dataframe(pd.links_external[:100])

    with st.expander("🤖 robots.txt & Sitemap"):
        st.markdown(f"**robots.txt:** {robots.robots_url}")
        st.json({
            "is_allowed": robots.is_allowed,
            "crawl_delay": robots.crawl_delay,
            "sitemaps": robots.sitemaps,
            "error": robots.error,
        })
        st.markdown("**Sitemap(s):**")
        st.json({
            "sitemap_urls": sitemap.sitemap_urls,
            "discovered_urls (sample)": sitemap.discovered_urls[:50],
            "fetched": sitemap.fetched,
            "error": sitemap.error,
        })

    with st.expander("🕷️ Crawled Pages (overview)"):
        overview = []
        for u, p in pages.items():
            overview.append({
                "url": u,
                "status": p.status,
                "title": p.title,
                "meta_desc_len": len(p.meta_description or ""),
                "h1_count": len(p.headings.get("h1", [])),
                "internal_links": len(p.links_internal),
                "external_links": len(p.links_external),
            })
        st.dataframe(overview)

    # Downloads
    st.subheader("⬇️ Export Report")
    export = {
        "summary": {
            "url": main_pd.final_url,
            "page_type": report.page_type,
            "scores": {
                "ai_readability": f"{report.ai_readability.value}/{report.ai_readability.out_of}",
                "seo": f"{report.seo.value}/{report.seo.out_of}",
                "content": f"{report.content.value}/{report.content.out_of}",
            },
        },
        "highlights": report.highlights,
        "issues": report.issues,
        "suggestions": report.suggestions,
        "raw": report.raw,
        "crawled_pages": {u: p.__dict__ for u, p in pages.items()},
    }
    buf = io.BytesIO(json.dumps(export, indent=2).encode("utf-8"))
    st.download_button("Download JSON", data=buf, file_name="ai_visibility_report.json", mime="application/json")

else:
    st.info("Enter a URL and click **Run Audit** to generate your AI Visibility Report.")
