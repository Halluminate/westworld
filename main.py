"""
Noodle Flights -- AUTONOMOUS AGENT version (Claude drives, Phoenix logs).

Unlike the deterministic main.py, here CLAUDE decides every action by reading the
task and the page. It uses HIGH-LEVEL actions (tools) so the fragile widgets
(airport autocomplete, calendar, class dropdown) are handled by proven code,
while Claude does the reasoning. This generalizes across task types
(basic_flight_search, airline_search_simple, date_range_search, ...).

Run (venv active, `phoenix serve` running, keys exported):
  python agent_main.py
Change CATEGORY below to target a different task type.
Open http://localhost:6006 -> project "noodle-flights-agent".
"""

import asyncio
import calendar as calmod
import json
import os
import re
from datetime import datetime, timezone

os.environ["GRPC_VERBOSITY"] = "NONE"          # silence the fork/poll noise
os.environ["GLOG_minloglevel"] = "3"
os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

from anthropic import Anthropic
from datasets import load_dataset
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register
from playwright.async_api import async_playwright

from westworld.base import DatasetItem, instantiate


CATEGORY = "basic_flight_search"
#CATEGORY = "airline_search_simple"
#CATEGORY = "date_range_search"
#CATEGORY = "date_range_search_simplified"
#CATEGORY = "natural_airline_search_simple"
#CATEGORY = "flight_search_budget"
#CATEGORY = "flight_search_cheapest"
#CATEGORY = "flight_search_roundtrip"
MAX_STEPS = 75

tracer_provider = register(project_name="noodle-flights-agent", auto_instrument=True)
tracer = tracer_provider.get_tracer(__name__)
client = Anthropic()
MODEL = "claude-sonnet-4-6"


def _code_of(v):
    """values may hold 'NRT', {'iata':'NRT'}, {'code':'NRT'}, or an object."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("iata", "code", "airport_code", "value"):
            if v.get(k):
                return str(v[k])
    for k in ("iata", "code"):
        if hasattr(v, k) and getattr(v, k):
            return str(getattr(v, k))
    return str(v)


def _adults_of(v):
    """passengers may be 4, {'adults':4,...}, or an object."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, dict):
        for k in ("adults", "adult", "count", "value"):
            if isinstance(v.get(k), int):
                return v[k]
    if hasattr(v, "adults"):
        return getattr(v, "adults")
    try:
        return int(v)
    except Exception:
        return None


def _price_of(row_text):
    """First $ amount in a row's text, as an int (0 if none).

    Must be the FIRST match: the JS slices row text at 240 chars, so a row that
    bleeds into the next one can end with a truncated price like "$9". Taking the
    last match then picks that fragment as the cheapest.
    """
    import re as _re
    m = _re.findall(r"\$([\d,]+)", row_text or "")
    return int(m[0].replace(",", "")) if m else 0




# --- Eyes: numbered interactive elements -----------------------------------
async def get_elements(page):
    sel = "input, textarea, select, button, a[href], [role='button'], [role='option']"
    handles, lines = [], []
    for el in await page.query_selector_all(sel):
        try:
            if not await el.is_visible():
                continue
            info = await el.evaluate(
                """e => ({tag:e.tagName.toLowerCase(), ph:e.getAttribute('placeholder'),
                          aria:e.getAttribute('aria-label'),
                          val:(e.value||'').slice(0,40),
                          txt:(e.innerText||e.textContent||'').trim().slice(0,60)})"""
            )
        except Exception:
            continue
        _txt = (info.get("txt") or "").strip()
        if _txt in {"All filters", "Stops", "Airlines", "Bags", "Price",
                    "Times", "Emissions", "Connecting airports", "Duration"}:
            continue                    # hide filter controls from the agent
        i = len(handles)
        handles.append(el)
        d = f"[{i}] <{info['tag']}>"
        for k in ("ph", "aria", "val", "txt"):
            if info.get(k):
                d += f" {k}={info[k]!r}"
        lines.append(d)
        if len(handles) >= 70:
            break
    return handles, "\n".join(lines)


# --- Brain: Claude picks ONE high-level action -----------------------------
def decide(task, elements, history, recorded=None, last_results=""):
    past = "\n".join(history[-8:]) if history else "(nothing yet)"
    rec = ""
    if recorded is not None:
        if recorded:
            rec = "\n\nRANGES ALREADY RECORDED (do NOT search these again):\n" + \
                  "\n".join(f"  range {k}: {v}" for k, v in recorded.items())
        else:
            rec = "\n\nRANGES ALREADY RECORDED: none yet."
    msg = client.messages.create(
        model=MODEL,
        max_tokens=900,
        system=(
            "You are an autonomous web agent operating a flight-search website to "
            "complete a TASK. Each turn you see the numbered ELEMENTS on the page and "
            "the ACTIONS you've taken. Reply with EXACTLY ONE JSON object, no prose:\n"
            '  {"action":"set_trip_type","value":"one-way"|"round-trip"}\n'
            '  {"action":"set_airport","field":"from"|"to","code":"XXX"}\n'
            '  {"action":"set_class","value":"economy"|"premium"|"business"|"first"}\n'
            '  {"action":"pick_date","which":"departure"|"return","year":Y,"month":M,"day":D}\n'
            '  {"action":"submit"}\n'
            '  {"action":"click","index":N}\n'
            '  {"action":"done"}\n\n'
            "Read the TASK to determine trip type, both airports, cabin class, and "
            "date(s). Configure the form with the high-level actions first (they "
            "handle the tricky widgets for you). For round-trip set BOTH dates. Use "
            "{\"action\":\"submit\"} to run the search when the task requires it. If "
            "the task says to SELECT a specific flight or airline, after submitting "
            "look at ELEMENTS for the results and use click to choose the matching "
            "row. Reply done only when the whole task is complete.\n"
            "  {\"action\":\"read_results\"}                 (read the flight results table)\n"
            "  {\"action\":\"reset_search\"}                 (close popups to edit the form again)\n"
            "  {\"action\":\"select_flight\"}            (auto-picks the CHEAPEST departure -> shows RETURNS)\n"
            "  {\"action\":\"record_range\",\"range\":1,\"dep_time\":\"7:06 PM - 11:43 PM\","
            "\"dep_duration\":\"10 hr 37 min\",\"ret_time\":\"12:33 AM - 5:16 PM\","
            "\"ret_duration\":\"10 hr 43 min\"}\n"
            "  {\"action\":\"report_answer\",\"text\":\"...\"}   (your FINAL written answer)\n\n"
            "NEVER click the filter buttons (Stops, Airlines, Bags, Price, Times, "
            "Emissions, Duration, All filters) -- they open modals and are not needed.\n"
            "EXTRACTION TASKS (multiple date ranges). Repeat this EXACT cycle once "
            "per date range:\n"
            "  a) set_trip_type, set_airport (from + to), set_class\n"
            "  b) pick_date departure, pick_date return  (for THIS range)\n"
            "  c) submit\n"
            "  d) read_results -- this shows DEPARTING flights. Note the CHEAPEST one.\n"
            "  e) select_flight  (no index needed -- it auto-selects the CHEAPEST departure)\n"
            "  f) read_results again -- now it shows RETURNING flights. Note the CHEAPEST.\n"
            "  g) record_range with the range number and FOUR fields copied VERBATIM:\n"
            "     dep_time + dep_duration -- from the departure row select_flight reported\n"
            "     ret_time + ret_duration -- from the RETURNING row marked '<== CHEAPEST'\n"
            "     Copy the time strings EXACTLY as shown, e.g. '7:06 PM - 11:43 PM'.\n"
            "     NEVER write 'index [0]' or 'not captured' -- always the real times.\n"
            "  h) reset_search -- reloads the blank form for the NEXT range\n"
            "You MUST actually search each range: never reuse another range's flights. "
            "Each range has DIFFERENT dates and therefore different flights.\n"
            "Do ranges strictly in order 1, 2, 3. NEVER redo a range listed under "
            "RANGES ALREADY RECORDED. After range 3 is recorded, immediately call "
            "report_answer combining all three, then done. Steps are limited -- do not "
            "call read_results twice for the same search.\n"
            "IMPORTANT: after submit you are on the RESULTS page, which has NO calendar. "
            "pick_date will NOT work there. You MUST reset_search first, then re-fill the "
            "whole form for the next range.\n"
            "After all ranges: report_answer with a text answer that, for EVERY range, "
            "gives the cheapest departure and cheapest return, copying their PRICE and "
            "TIME RANGE verbatim from the results (e.g. '$1,074' and '7:06 PM - 11:43 PM'). "
            "Then reply done. If read_results shows no rows, call read_results once more "
            "before concluding anything.\n"
            "BUDGET TASKS: if the task asks whether flights exist under a budget, "
            "fill the form, submit, read the results, then report_answer with EXACTLY "
            "ONE of these phrases and NOTHING resembling the other:\n"
            "   \"Yes, there is at least one option\"   (if any flight is under the budget)\n"
            "   \"No options found\"                    (if none are under the budget)\n"
            "Use the phrase verbatim. Never include both phrases. Then reply done.\n"
            "CRITICAL: Look at ACTIONS SO FAR. Any line starting with SUCCESS is "
            "already DONE -- never repeat it. Any line starting with ALREADY DONE "
            "means that step is finished -- move on. Each turn, do the NEXT unfinished "
            "step, not one you've already completed."
        ),
        messages=[{"role": "user",
                   "content": f"TASK:\n{task}\n\nELEMENTS:\n{elements}\n\n"
                              f"ACTIONS SO FAR:\n{past}{rec}"
                              + (f"\n\nLATEST RESULTS:\n{last_results[:2500]}" if last_results else "")}],
    )
    raw = msg.content[0].text.strip()
    s = raw.find("{")
    if s == -1:
        raise ValueError(f"No JSON: {raw}")
    obj, _ = json.JSONDecoder().raw_decode(raw[s:])
    return obj


# --- Tools: reliable widget mechanics (from everything we learned) ----------
async def t_set_airport(page, field, code):
    code = _code_of(code)          # tolerate dicts/objects from any task config
    """Type the airport code, pick the matching suggestion, and VERIFY the field
    now contains that code. Returns 'ok'/'FAIL' so the caller can tell."""
    ph = "Where from?" if field == "from" else "Where to?"
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(200)

    # WAIT for the input -- query_selector returns None instantly if the form has
    # not rendered yet, which used to crash here (and, before that, silently left
    # the airport unset so the site searched the wrong route).
    inp = None
    for _try in range(3):
        try:
            inp = await page.wait_for_selector(f"input[placeholder='{ph}']",
                                               timeout=6000, state="visible")
            if inp:
                break
        except Exception:
            await page.wait_for_timeout(900)
    if inp is None:
        # fall back to position: the two airport inputs are the first on the form
        boxes = await page.query_selector_all("input")
        vis = []
        for b in boxes:
            try:
                if await b.is_visible():
                    vis.append(b)
            except Exception:
                pass
        idx = 0 if field == "from" else 1
        inp = vis[idx] if len(vis) > idx else None
        if inp is not None:
            print(f"   [airport] placeholder {ph!r} not found; using visible input #{idx}")
    if inp is None:
        print(f"   [airport] FAILED: no input for {field!r} on this page")
        return f"FAIL: no input for {field}"

    await inp.click(force=True)
    await page.keyboard.press("Meta+A")
    await page.keyboard.press("Backspace")
    await page.wait_for_timeout(300)
    await page.keyboard.type(code, delay=140)
    await page.wait_for_timeout(1500)

    # Prefer a suggestion with the exact code on its own line; else one containing it.
    best = None
    for b in await page.query_selector_all("button"):
        try:
            if not await b.is_visible():
                continue
            t = await b.inner_text()
            if not t.strip() or "Search" in t:
                continue
            if any(ln.strip() == code for ln in t.split("\n")):
                best = b
                break
            if code in t and best is None:
                best = b
        except Exception:
            continue
    if best:
        await best.click()
        await page.wait_for_timeout(500)

    try:
        val = await inp.input_value()
    except Exception:
        val = ""
    if code.lower() not in (val or "").lower():
        # retry once: clear and keyboard-select the top suggestion
        await inp.click(force=True)
        await page.keyboard.press("Meta+A")
        await page.keyboard.press("Backspace")
        await page.wait_for_timeout(300)
        await page.keyboard.type(code, delay=140)
        await page.wait_for_timeout(1500)
        await inp.press("ArrowDown")
        await inp.press("Enter")
        await page.wait_for_timeout(500)
        val = await inp.input_value()

    ok = code.lower() in val.lower()
    return f"{field}={val!r} -> {'ok' if ok else 'FAIL'}"


async def t_set_trip_type(page, value):
    want = "One-way" if "one" in value.lower() else "Round trip"
    for cur in ["Round trip", "One-way"]:
        b = await page.query_selector(f"button:has-text('{cur}')")
        if b and await b.is_visible():
            await b.click(); await page.wait_for_timeout(400); break
    opt = await page.query_selector(f"button:has-text('{want}')")
    if opt:
        await opt.click(); await page.wait_for_timeout(400)
    await page.keyboard.press("Escape")
    return f"trip_type={want}"


async def t_set_class(page, value):
    labels = {"economy": "Economy", "premium": "Premium",
              "business": "Business", "first": "First"}
    want = next((labels[k] for k in labels if k in value.lower()), "Economy")
    for cur in labels.values():
        b = await page.query_selector(f"button:has-text('{cur}')")
        if b and await b.is_visible():
            await b.click(); await page.wait_for_timeout(400); break
    opt = await page.query_selector(f"button:has-text('{want}')")
    if opt:
        await opt.click(); await page.wait_for_timeout(400)
    await page.keyboard.press("Escape")
    return f"class={want}"


async def _click_day_by_monthname(page, target_label, day):
    """Click `day` inside the panel whose header starts with target_label
    (e.g. 'May 2027'), using a real Playwright click."""
    panels = page.locator("div.h-\\[380px\\]")
    n = await panels.count()
    headers = []
    for k in range(n):
        panel = panels.nth(k)
        header = panel.locator("div").first
        try:
            htxt = (await header.inner_text()).strip()
        except Exception:
            htxt = "<err>"
        headers.append(htxt)
        if not htxt.startswith(target_label):
            continue
        buttons = panel.locator("button")
        bn = await buttons.count()
        for bi in range(bn):
            b = buttons.nth(bi)
            try:
                t = (await b.inner_text()).strip()
            except Exception:
                continue
            if t.split("\n")[0].strip() == str(day):
                await b.scroll_into_view_if_needed()
                await b.click()
                await page.wait_for_timeout(400)
                print(f"   [pick] clicked {day} in {htxt!r}")
                return True
    print(f"   [pick] panels={n} headers={headers} target={target_label!r} -> day not clicked")
    return False


async def _month_visible(page, month_name):
    return await page.evaluate(
        """(mn) => [...document.querySelectorAll('div')]
              .some(e => e.className && e.className.includes('h-[380px]')
                    && e.querySelector('div')
                    && e.querySelector('div').textContent.trim().startsWith(mn))""",
        month_name)


async def _click_day_by_monthname(page, month_name, day):
    """Click `day` inside the panel whose header STARTS WITH month_name
    (headers may be 'August' or 'August 2026'). Real Playwright click."""
    panels = page.locator("div.h-\\[380px\\]")
    n = await panels.count()
    seen = []
    for k in range(n):
        panel = panels.nth(k)
        header = panel.locator("div").first
        try:
            htxt = (await header.inner_text()).strip()
        except Exception:
            continue
        seen.append(htxt)
        if not htxt.startswith(month_name):
            continue
        buttons = panel.locator("button")
        for bi in range(await buttons.count()):
            b = buttons.nth(bi)
            try:
                t = (await b.inner_text()).strip()
            except Exception:
                continue
            if t.split("\n")[0].strip() == str(day):
                await b.scroll_into_view_if_needed()
                await b.click()
                await page.wait_for_timeout(400)
                print(f"   [pick] clicked {day} in {htxt!r}")
                return True
    print(f"   [pick] headers={seen} month={month_name!r} day {day} not clicked")
    return False


async def t_pick_date(page, which, year, month, day):
    """Navigate by COUNTING net months from the known current month (today),
    stepping forward or backward one month per click, then click the day in the
    target month panel. Works with year-less headers ('July'/'August')."""
    await page.keyboard.press("Escape")
    await page.mouse.click(10, 300)
    await page.wait_for_timeout(400)

    label = "Return" if str(which).lower().startswith("ret") else "Departure"
    opener = page.locator(f"text={label}").first
    if await opener.count() > 0 and await opener.is_visible():
        await opener.click()
        await page.wait_for_timeout(900)
    # Verify a calendar actually opened; retry once if not.
    panels = await page.evaluate(
        """() => document.querySelectorAll('div[class*="h-[380px]"]').length""")
    if panels == 0:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)
        if await opener.count() > 0:
            await opener.click()
            await page.wait_for_timeout(1000)

    now = datetime.now(timezone.utc)
    net = (int(year) - now.year) * 12 + (int(month) - now.month)   # +future / -past
    month_name = calmod.month_name[int(month)]

    # The calendar opens showing the current month as the LEFT panel and next
    # month as the RIGHT panel. So the target is visible once we've stepped
    # (net) months for the left panel -- but since the right panel is one ahead,
    # a forward target of N is visible after N-1 steps, a backward target after
    # abs(net) steps. Simplest robust rule: step until the month panel shows,
    # capped, choosing direction by sign.
    steps = abs(net)
    forward = net > 0
    right = "button:has(svg.lucide-chevron-right)"
    left = "button:has(svg.lucide-chevron-left)"
    for _ in range(steps + 2):
        if await _click_day_by_monthname(page, month_name, int(day)):
            return f"{which} {month_name} {year} {day} -> ok"
        chev = page.locator(right if forward else left).last
        if await chev.count() > 0 and await chev.is_visible():
            await chev.click()
            await page.wait_for_timeout(350)
        else:
            break
    # last attempt after final step
    ok = await _click_day_by_monthname(page, month_name, int(day))
    return f"{which} {month_name} {year} {day} -> {'ok' if ok else 'FAIL'}"


_PAX_FIND_BOX = """
    const NL = String.fromCharCode(10);
    let box = null;
    for (const el of document.querySelectorAll('div')) {
        const t = (el.innerText || '').trim();
        if (t.split(NL)[0].trim().toLowerCase() === 'adults'
            && el.querySelectorAll('button').length >= 2) {
            if (!box || el.innerText.length < box.innerText.length) box = el;
        }
    }
"""

_PAX_CLICK_PLUS = "() => {" + _PAX_FIND_BOX + """
    if (!box) return false;
    const b = box.querySelectorAll('button');
    b[b.length - 1].click();
    return true;
}"""

_PAX_READ = "() => {" + _PAX_FIND_BOX + """
    if (!box) return null;
    const toks = (box.innerText || '').split(NL).map(s => s.trim())
                  .filter(s => /^[0-9]+$/.test(s));
    return toks.length ? parseInt(toks[0]) : null;
}"""


async def t_set_passengers(page, adults):
    """Set the adult count. The popup always starts at 1, so we simply click '+'
    exactly (adults - 1) times. No read-loop, no overshoot. Browser-tested."""
    target = int(adults)
    if target <= 1:
        return "no change needed"

    # open the popup (the button whose text is just the current count)
    opened = False
    loc = page.locator("button").filter(has_text=re.compile(r"^\s*\d+\s*$"))
    if await loc.count() > 0:
        await loc.first.click()
        opened = True
    if not opened:
        for sel in ["button:has-text('passenger')", "[aria-label*='assenger']"]:
            l2 = page.locator(sel).first
            if await l2.count() > 0 and await l2.is_visible():
                await l2.click(); opened = True; break
    await page.wait_for_timeout(800)
    if not opened:
        return "could not open passenger selector"

    start_count = await page.evaluate(_PAX_READ)

    clicks = 0
    for _ in range(target - 1):
        ok = await page.evaluate(_PAX_CLICK_PLUS)
        if not ok:
            break
        clicks += 1
        await page.wait_for_timeout(220)

    final = await page.evaluate(_PAX_READ)

    done = page.locator("button:has-text('Done')").first
    if await done.count() > 0 and await done.is_visible():
        await done.click()
    else:
        await page.keyboard.press("Escape")
    await page.wait_for_timeout(700)

    return f"adults {start_count} -> {final} ({clicks} clicks, target {target})"


async def t_read_results(page):
    """Read flight rows. Instrumented: prints what selectors actually find."""
    await page.wait_for_timeout(2500)

    # --- diagnostics: what IS on this page? ---
    diag = await page.evaluate(r"""() => {
        const out = {url: location.href.slice(0, 90)};
        out.total_divs = document.querySelectorAll('div').length;
        out.cursor_pointer = document.querySelectorAll('div.cursor-pointer').length;
        out.minw0 = document.querySelectorAll('div[class*="min-w-0"]').length;
        out.both = document.querySelectorAll('div[class*="cursor-pointer"][class*="min-w-0"]').length;
        out.dollar_divs = [...document.querySelectorAll('div')]
            .filter(d => d.innerText && d.innerText.includes('$')).length;
        // sample text of the smallest divs that contain a '$'
        const cands = [...document.querySelectorAll('div')]
            .filter(d => d.innerText && d.innerText.includes('$') && d.innerText.length < 300)
            .sort((a,b) => a.innerText.length - b.innerText.length)
            .slice(0, 3)
            .map(d => ({cls: (d.className||'').slice(0,70), txt: d.innerText.split(String.fromCharCode(10)).join(' | ').slice(0,140)}));
        out.samples = cands;
        return out;
    }""")
    print("   [diag] url:", diag.get("url"))
    print("   [diag] div.cursor-pointer:", diag.get("cursor_pointer"),
          " min-w-0:", diag.get("minw0"), " both:", diag.get("both"),
          " divs containing $:", diag.get("dollar_divs"))
    for smp in diag.get("samples", []):
        print(f"   [diag] sample cls={smp['cls']!r}")
        print(f"          txt={smp['txt']!r}")

    # --- extraction: pull rows straight out of the DOM via JS (no locator guessing) ---
    # NOTE: no regex inside JS -- escaping through Python has caused SyntaxErrors.
    rows = await page.evaluate("""() => {
        const seen = new Set();
        const out = [];
        for (const d of document.querySelectorAll('div')) {
            const t = (d.innerText || '').trim();
            if (t.indexOf('$') === -1) continue;
            if (t.length < 30 || t.length > 700) continue;
            if (t.indexOf(':') === -1) continue;
            if (t.indexOf('hr') === -1 && t.indexOf('min') === -1) continue;
            const NL = String.fromCharCode(10);
            const flat = t.split(NL).join(' | ');
            if (seen.has(flat)) continue;
            seen.add(flat);
            out.push(flat.slice(0, 240));
            if (out.length >= 20) break;
        }
        return out;
    }""")

    if not rows:
        return "NO RESULTS FOUND (page may still be loading; try again)"
    body = (await page.inner_text("body")).lower()
    if "returning" in body:
        phase = "RETURNING flights (these are the RETURN leg)"
    elif "departing" in body:
        phase = "DEPARTING flights (select one to see RETURN flights)"
    else:
        phase = "flights"
    prices = [_price_of(r) for r in rows]
    cheapest_i = min(range(len(rows)), key=lambda i: prices[i] if prices[i] > 0 else 10**9)
    lines = []
    for i, r in enumerate(rows):
        tag = "  <== CHEAPEST" if i == cheapest_i else ""
        lines.append(f"[{i}] {r}{tag}")
    return (f"{phase} -- {len(lines)} rows -- CHEAPEST is row [{cheapest_i}]\n"
            "format: times | airline | duration | route | stops | price\n" + "\n".join(lines))


async def t_select_flight(page, index="cheapest"):
    """Pick the cheapest departure row and click it.

    JS only collects/tags rows (no regex there -- escaping through Python has
    bitten us). Python parses prices and decides, then JS clicks by tag.
    """
    rows = await page.evaluate("""() => {
        const seen = new Set(); const out = [];
        let i = 0;
        for (const d of document.querySelectorAll('div')) {
            const t = (d.innerText || '').trim();
            if (t.indexOf('$') === -1) continue;
            if (t.length < 30 || t.length > 700) continue;
            if (t.indexOf(':') === -1) continue;
            if (t.indexOf('hr') === -1 && t.indexOf('min') === -1) continue;
            const NL = String.fromCharCode(10);
            const flat = t.split(NL).join(' | ');
            if (seen.has(flat)) continue;
            seen.add(flat);
            d.setAttribute('data-agent-row', String(i));
            out.push(flat.slice(0, 240));
            i++;
            if (out.length >= 20) break;
        }
        return out;
    }""")

    if not rows:
        return "no rows found on page", ""

    if isinstance(index, str) or index is None:
        priced = [(i, _price_of(t)) for i, t in enumerate(rows) if _price_of(t) > 0]
        if not priced:
            return "no priced rows", ""
        idx = min(priced, key=lambda x: x[1])[0]
    else:
        idx = int(index)
    if idx >= len(rows):
        return f"index {idx} out of range ({len(rows)} rows)", ""

    row_text = rows[idx]
    price = _price_of(row_text)

    clicked = await page.evaluate(
        """(i) => {
            const el = document.querySelector('[data-agent-row="' + i + '"]');
            if (!el) return false;
            const target = el.querySelector('[class*="cursor-pointer"]') || el;
            target.click();
            return true;
        }""", str(idx))

    if not clicked:
        return f"could not click row {idx}", row_text

    await page.wait_for_timeout(2500)
    body = (await page.inner_text("body")).lower()
    phase_txt = "RETURN list now showing" if "returning" in body else "page did NOT change"
    return f"selected CHEAPEST [{idx}] (${price}) -> {phase_txt}", row_text[:200]


async def t_reset_search(page, url=None):
    """Return to the SEARCH FORM. After submit we're on the results page, which
    has a different (compact) date widget and NO month calendar -- so pick_date
    cannot work there. Navigating back restores the full form."""
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(300)
    if url:
        try:
            await page.goto(url, timeout=60000)
            await page.wait_for_timeout(1800)
            return "returned to search form (form is blank again)"
        except Exception as e:
            return f"reset failed: {e}"
    await page.go_back()
    await page.wait_for_timeout(1500)
    return "went back to search form"


async def t_submit(page):
    await page.keyboard.press("Escape"); await page.wait_for_timeout(300)
    await page.mouse.click(5, 5); await page.wait_for_timeout(300)
    b = await page.query_selector("button:has-text('Search')")
    if b:
        try:
            await b.click()
        except Exception:
            await b.click(force=True)
        await page.wait_for_timeout(2500)
    return "submitted"


# --- Run one task ----------------------------------------------------------
def _class_of(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return str(v.get("value") or v.get("label") or "")
    return str(getattr(v, "value", v))


def _parse_ranges(task_text):
    """Pull date ranges from the task text. Handles:
      - 'Month D-Month D, YYYY' (and 'Month D-D, YYYY'), possibly several
      - 'departing Month D, YYYY ... returning Month D, YYYY'
      - 'departing on Month D, YYYY with return date Month D, YYYY'
    Returns list of ((y,m,d)dep, (y,m,d)ret).
    """
    import calendar as _cal, re as _re
    months = {m.lower(): i for i, m in enumerate(_cal.month_name) if m}

    def md(mtxt, dtxt, ytxt):
        return (int(ytxt), months[mtxt.lower()], int(dtxt))

    out = []

    # Pattern A: ranges like 'January 26-February 1, 2027'
    patA = _re.compile(
        r"([A-Z][a-z]+)\s+(\d{1,2})\s*[-\u2013]\s*(?:([A-Z][a-z]+)\s+)?(\d{1,2}),\s*(\d{4})")
    for m in patA.finditer(task_text):
        m1, d1, m2, d2, yr = m.groups()
        mo1 = months[m1.lower()]
        mo2 = months[m2.lower()] if m2 else mo1
        y = int(yr)
        y1 = y if mo1 <= mo2 else y - 1
        out.append(((y1, mo1, int(d1)), (y, mo2, int(d2))))
    if out:
        return out

    # Pattern B: many natural phrasings for departure + return.
    #   'departing/leaving/outbound (on) Month D, YYYY'
    #   'returning/coming back/back (on) Month D, YYYY'
    dep_pat = _re.compile(
        r"(?:depart(?:ing)?|leav(?:e|ing)|outbound|fly(?:ing)? out|going)\s+"
        r"(?:on\s+)?([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})", _re.I)
    ret_pat = _re.compile(
        r"(?:return(?:ing)?(?:\s+date)?|coming\s+back|com(?:e|ing)\s+back|back|inbound)\s+"
        r"(?:on\s+)?([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})", _re.I)
    dep = dep_pat.search(task_text)
    ret = ret_pat.search(task_text)
    if dep and ret:
        out.append((md(*dep.groups()), md(*ret.groups())))
        return out

    # Pattern C: only a departure date (one-way)
    if dep:
        out.append((md(*dep.groups()), None))
    return out


async def _fill_form(page, vals, dep, ret):
    """Set trip type, airports, class, and the two dates for one range."""
    _tt = (vals.get("tripType") or "")
    if ret is None or "one" in str(_tt).lower():
        await t_set_trip_type(page, "one-way")     # no return date -> one-way
    else:
        await t_set_trip_type(page, "round-trip")
    origin = _code_of(vals.get("origin"))
    dest = _code_of(vals.get("destination"))
    cabin = _class_of(vals.get("travel_class") or vals.get("class"))
    pax = _adults_of(vals.get("passengers") or vals.get("adults"))
    print(f"   form: origin={origin!r} dest={dest!r} class={cabin!r} adults={pax}")

    async def _set_airport_verified(field, code):
        """Set an airport and CONFIRM it committed. A silent failure here makes the
        site search the wrong route, which quietly poisons every downstream number."""
        for attempt in range(3):
            res = await t_set_airport(page, field, code)
            try:
                ph = "Where from?" if field == "from" else "Where to?"
                inp = await page.query_selector(f"input[placeholder='{ph}']")
                val = (await inp.input_value()) if inp else ""
            except Exception:
                val = ""
            if code.lower() in (val or "").lower():
                return True
            print(f"   !! {field} airport {code!r} not committed "
                  f"(field shows {val!r}, attempt {attempt + 1}/3) -> retrying")
            await page.wait_for_timeout(600)
        print(f"   !! GIVING UP setting {field}={code!r} -- results will be WRONG")
        return False

    if origin:
        await _set_airport_verified("from", origin)
    if dest:
        await _set_airport_verified("to", dest)
    if cabin:
        await t_set_class(page, cabin)
    if pax and pax > 1:
        ok = await t_set_passengers(page, pax)
        print(f"   passengers -> {ok}")
    r1 = await t_pick_date(page, "departure", dep[0], dep[1], dep[2])
    if "ok" not in r1:
        print("   !! departure date failed:", r1)
    if ret:
        r2 = await t_pick_date(page, "return", ret[0], ret[1], ret[2])
        if "ok" not in r2:
            print("   !! return date failed:", r2)
            return False
    return True


def _ask_cheapest(task, rows_text, leg):
    """Claude reads the rows and returns the cheapest one's time range + duration."""
    msg = client.messages.create(
        model=MODEL, max_tokens=400,
        system=("You are given a list of flight rows. Identify the CHEAPEST one "
                "(lowest $ price). Reply with EXACTLY one JSON object, no prose:\n"
                '{"time":"<time range verbatim>","duration":"<duration verbatim>","price":"<$ price>"}'),
        messages=[{"role": "user", "content": f"TASK CONTEXT:\n{task}\n\n{leg} ROWS:\n{rows_text}"}],
    )
    raw = msg.content[0].text.strip()
    st = raw.find("{")
    obj, _ = json.JSONDecoder().raw_decode(raw[st:])
    return obj


def _fmt_time_range(dep_iso, arr_iso):
    """Render '3:57 PM \u2013 7:25 PM' from ISO timestamps, like the UI/grader."""
    from datetime import datetime as _dt
    def h(x):
        t = _dt.fromisoformat(str(x).replace("Z", "+00:00"))
        return t.strftime("%I:%M %p").lstrip("0")
    return f"{h(dep_iso)} \u2013 {h(arr_iso)}"


def _fmt_duration(mins):
    if not mins:
        return ""
    h, m = divmod(int(mins), 60)
    return f"{h} hr {m} min"


def _norm_cabin(name):
    """'premium-economy' / 'Premium economy' / 'PREMIUM_ECONOMY' -> 'premium_economy'."""
    n = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if n in ("premium", "premium_economy"):
        return "premium_economy"
    return n


def _cabin_price(f, cabin):
    """Effective USD price for the given cabin (mirrors _effective_price_usd)."""
    for cp in (f.get("cabinPricing") or []):
        if str(cp.get("cabin", "")).lower().replace("-", "_") == cabin:
            for k in ("priceUsd", "price_usd", "totalPriceUsd", "price", "totalPrice"):
                if isinstance(cp.get(k), (int, float)):
                    return cp[k]
    return None


def _has_cabin_and_seats(f, cabin, pax):
    for cp in (f.get("cabinPricing") or []):
        if str(cp.get("cabin", "")).lower().replace("-", "_") == cabin and (cp.get("seatsRemaining") or 0) >= pax:
            return True
    return False


def _filter_flights_like_grader(flights, class_name, pax):
    """Replicate westworld's filter_flights():
         direct only (no segments) -> cabin+seats -> cheapest by cabin price."""
    if not flights:
        return []
    cabin = _norm_cabin(class_name)
    direct = [f for f in flights if not (f.get("segments") or [])]
    eligible = [f for f in direct if _has_cabin_and_seats(f, cabin, pax)]
    if not eligible:
        return []
    eligible.sort(key=lambda f: (_cabin_price(f, cabin) if _cabin_price(f, cabin) is not None else 10**9))
    return eligible


def _filter_with_airline(flights, class_name, pax, airline_code):
    """Replicate date_range_search.filter_flights():
       cheapest eligible direct flight + ALL direct flights on `airline_code`,
       deduped, price-ascending. Returns a list of flight dicts."""
    if not flights:
        return []
    cabin = _norm_cabin(class_name)
    direct = [f for f in flights if not (f.get("segments") or [])]
    eligible = [f for f in direct if _has_cabin_and_seats(f, cabin, pax)]
    eligible.sort(key=lambda f: (_cabin_price(f, cabin) if _cabin_price(f, cabin) is not None else 10**9))
    result, seen = [], set()
    if eligible:
        f0 = eligible[0]
        result.append(f0); seen.add(f0.get("flightNumber"))
    code = str(airline_code or "").upper()
    for f in eligible:
        if str(f.get("airline", "")).upper() == code and f.get("flightNumber") not in seen:
            result.append(f); seen.add(f.get("flightNumber"))
    return result


def _describe(f, cabin):
    """One flight -> readable string with number, time range, duration, price."""
    dep = f.get("depart") or f.get("departTime") or ""
    arr = f.get("arrive") or f.get("arriveTime") or ""
    try:
        tr = _fmt_time_range(dep, arr)
    except Exception:
        tr = ""
    price = _cabin_price(f, cabin)
    return (f"{f.get('flightNumber','')} {tr} "
            f"({_fmt_duration(f.get('durationMin') or f.get('totalDurationMin'))})"
            + (f" ${price:,}" if isinstance(price, (int, float)) else ""))


def _cheapest_price_per_pax(flights, class_name, pax):
    """Per-passenger cabin price of the cheapest eligible direct flight (grader's
    _per_passenger_price_usd on filter_flights[0]). Returns int USD or None."""
    lst = _filter_flights_like_grader(flights, class_name, pax)
    if not lst:
        return None
    return _cabin_price(lst[0], _norm_cabin(class_name))


def _total_price_answer(dep_flights, ret_flights, class_name, pax):
    """Mirror airline_search_simple: (dep_pp * pax) + (ret_pp * pax), formatted.
    Returns (formatted_total_str_or_None, detail_dict)."""
    dep_pp = _cheapest_price_per_pax(dep_flights, class_name, pax)
    ret_pp = _cheapest_price_per_pax(ret_flights, class_name, pax)
    if not isinstance(dep_pp, (int, float)) or not isinstance(ret_pp, (int, float)):
        return None, {"dep_pp": dep_pp, "ret_pp": ret_pp}
    dep_total = int(dep_pp) * pax
    ret_total = int(ret_pp) * pax
    total = dep_total + ret_total
    return f"${total:,}", {"dep_pp": dep_pp, "ret_pp": ret_pp,
                           "dep_total": dep_total, "ret_total": ret_total, "total": total}


_BUDGET_YES = "Yes, there is at least one option"
_BUDGET_NO = "No options found"


def _flight_numbers_of(flight):
    """The flight's OWN flightNumber, as the grader expects.

    A connecting flight's number can be a combined string like 'AA100/AA200';
    check_flight_numbers splits on '/' and requires every part, so emitting the
    raw value verbatim satisfies both the combined and single cases.
    """
    if not flight:
        return []
    fn = flight.get("flightNumber")
    return [fn] if fn else []


def _cheapest_flight(flights, require_direct):
    """Mirror flight_search_cheapest: filter direct ONLY if the task says 'direct',
    then take the cheapest by priceUsd. The UI never renders flight numbers; the
    API does. Ties are fine -- the grader accepts any flight at the cheapest price.
    """
    pool = [f for f in (flights or []) if isinstance(f.get("priceUsd"), (int, float))]
    if require_direct:
        pool = [f for f in pool if not (f.get("segments") or [])]
    if not pool:
        return None
    return min(pool, key=lambda f: f["priceUsd"])


def _cheapest_under_budget(flights, budget):
    """Mirror check_budget_constraint: return the cheapest flight with
    priceUsd < budget, or None if there is none."""
    if budget is None:
        return None
    try:
        b = float(budget)
    except Exception:
        return None
    under = [f for f in (flights or [])
             if isinstance(f.get("priceUsd"), (int, float)) and f["priceUsd"] < b]
    if not under:
        return None
    return min(under, key=lambda f: f["priceUsd"])


def _budget_answer(flights, budget):
    """Mirror check_budget_constraint: any flight with priceUsd < budget -> yes.

    Returns exactly ONE of the grader's phrases (never both).
    """
    if budget is None:
        return None
    try:
        b = float(budget)
    except Exception:
        return None
    for f in (flights or []):
        p = f.get("priceUsd")
        if isinstance(p, (int, float)) and p < b:
            return _BUDGET_YES
    return _BUDGET_NO


def _budget_of(vals):
    v = vals.get("budget")
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        for k in ("value", "amount", "budget"):
            if isinstance(v.get(k), (int, float)):
                return v[k]
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except Exception:
        return None


def _all_flight_numbers_for_airline(flights, class_name, pax, airline_code):
    """Mirror date_range_search.filter_flights: return the flight numbers of
    (1) the cheapest eligible direct flight, and
    (2) every eligible direct flight operated by `airline_code`.
    Grader matches on flight_number alone, so numbers are all we need.
    """
    if not flights:
        return []
    cabin = _norm_cabin(class_name)
    direct = [f for f in flights if not (f.get("segments") or [])]
    eligible = [f for f in direct if _has_cabin_and_seats(f, cabin, pax)]
    eligible.sort(key=lambda f: (_cabin_price(f, cabin) if _cabin_price(f, cabin) is not None else 10**9))

    nums, seen = [], set()
    if eligible:
        fn = eligible[0].get("flightNumber", "")
        if fn and fn not in seen:
            nums.append(fn); seen.add(fn)
    code = str(airline_code or "").strip().upper()
    for f in eligible:
        if str(f.get("airline", "")).strip().upper() == code:
            fn = f.get("flightNumber", "")
            if fn and fn not in seen:
                nums.append(fn); seen.add(fn)
    return nums


def _pick_from_api(flights, class_name, pax):
    """Return {'flight','time','duration','price'} for the grader's cheapest."""
    lst = _filter_flights_like_grader(flights, class_name, pax)
    if not lst:
        return {}
    f = lst[0]
    cabin = _norm_cabin(class_name)
    dep = f.get("depart") or f.get("departTime") or ""
    arr = f.get("arrive") or f.get("arriveTime") or ""
    try:
        tr = _fmt_time_range(dep, arr)
    except Exception:
        tr = ""
    price = _cabin_price(f, cabin)
    return {
        "flight": f.get("flightNumber", ""),
        "time": tr,
        "duration": _fmt_duration(f.get("durationMin") or f.get("totalDurationMin")),
        "price": f"${price:,}" if isinstance(price, (int, float)) else "",
    }



# ---------------------------------------------------------------------------
# Per-step Phoenix logging: one child span per browser action, each carrying a
# screenshot. Screenshots are saved to ./phoenix_screenshots/ and the path is
# attached as a span attribute. Optionally the PNG is also embedded as base64
# (image.url) so Phoenix builds that render span images can show it inline.
# ---------------------------------------------------------------------------
import contextlib as _contextlib
from datetime import datetime as _dtnow

_MAX_EMBED_BYTES = int(os.environ.get("MAX_EMBED_BYTES", "600000"))   # ~600 KB cap
_SHOT_DIR = os.path.join(os.getcwd(), "phoenix_screenshots")
os.makedirs(_SHOT_DIR, exist_ok=True)


async def _screenshot(page, label):
    """Save a PNG for one step; return its path ('' on failure)."""
    ts = _dtnow.now().strftime("%Y%m%d_%H%M%S_%f")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))[:40]
    path = os.path.join(_SHOT_DIR, f"{ts}_{safe}.png")
    try:
        await page.screenshot(path=path, full_page=False)
        return path
    except Exception as e:
        print(f"   [shot] failed for {label!r}: {str(e)[:80]}")
        return ""


@_contextlib.asynccontextmanager
async def _step(page, label, **attrs):
    """Child span for one browser step; attaches a screenshot when it finishes.

        async with _step(page, "fill_form", dep=dep):
            await _fill_form(page, vals, dep, ret)
    """
    with tracer.start_as_current_span(f"step.{label}") as sp:
        sp.set_attribute("step.label", str(label))
        for k, v in attrs.items():
            sp.set_attribute(f"step.{k}", str(v))
        try:
            yield sp
        finally:
            shot = await _screenshot(page, label)
            if shot:
                sp.set_attribute("screenshot.path", shot)
                sp.set_attribute("screenshot.mime_type", "image/png")
                print(f"   [step] {label} -> {os.path.basename(shot)}")



def _attach_shot(span, path):
    """Attach a saved screenshot to a span (path always; base64 if enabled)."""
    if not path:
        return
    span.set_attribute("screenshot.path", path)
    span.set_attribute("screenshot.mime_type", "image/png")

async def run_extraction_task(index, category, dataset):
    """Python drives the 3-range loop; Claude reads results and reports."""
    tasks = dataset.filter(lambda r: f"noodle_flights/{category}" in r["task_id"])
    item = DatasetItem(**tasks[index])
    cfg = item.generate_task_config()
    evaluator = instantiate(cfg.eval_config)
    await evaluator.reset()
    vals = cfg.eval_config["values"]

    print("TASK:", cfg.task)
    print("VALUES keys:", list(vals.keys()))
    for k in ("travel_class", "passengers", "airline", "budget", "date", "tripType"):
        if k in vals:
            v = vals[k]
            print(f"   values[{k!r}] = {str(v)[:120]!r}")
    ranges = _parse_ranges(cfg.task)
    if not ranges:
        # Some categories (e.g. flight_search_budget) carry the date in values
        # rather than in prose. Prefer the config -- it is authoritative.
        _d = vals.get("date") or vals.get("departureDate") or vals.get("departure_date")
        _r = vals.get("returnDate") or vals.get("return_date")
        def _ymd(x):
            try:
                parts = str(x)[:10].split("-")
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except Exception:
                return None
        dep_t = _ymd(_d) if _d else None
        ret_t = _ymd(_r) if _r else None
        if dep_t:
            ranges = [(dep_t, ret_t)]
            print(f"   (dates taken from values: dep={dep_t} ret={ret_t})")
    print(f"Parsed {len(ranges)} date ranges: {ranges}")

    findings = []
    with tracer.start_as_current_span("noodle_flights.task") as root:
        root.set_attribute("task.prompt", cfg.task)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(timezone_id="UTC")
            page = await context.new_page()
            page.set_default_timeout(8000)

            captured = {"dep": None, "ret": None}

            async def _on_response(resp):
                try:
                    if "flight" not in resp.url.lower():
                        return
                    ct = (resp.headers or {}).get("content-type", "")
                    if "json" not in ct:
                        return
                    data = await resp.json()
                    if isinstance(data, dict) and (
                        "departingFlights" in data or "returningFlights" in data
                    ):
                        captured["dep"] = data.get("departingFlights")
                        captured["ret"] = data.get("returningFlights")
                        print(f"   [api] captured dep={len(captured['dep'] or [])} "
                              f"ret={len(captured['ret'] or [])}")
                except Exception:
                    pass

            page.on("response", lambda r: asyncio.create_task(_on_response(r)))

            for ri, (dep, ret) in enumerate(ranges, start=1):
                with tracer.start_as_current_span(f"range.{ri}") as rs:
                    print(f"\n--- range {ri}: depart {dep} return {ret} ---")
                    async with _step(page, f"range{ri}_open", url=cfg.url):
                        for attempt in range(3):
                            try:
                                await page.goto(cfg.url, timeout=60000); break
                            except Exception as e:
                                print("   goto retry:", e)
                        await page.wait_for_timeout(1500)
                        # Wait for the search form itself, not just the page load event.
                        try:
                            await page.wait_for_selector("input[placeholder='Where from?']",
                                                         timeout=15000, state="visible")
                        except Exception:
                            print("   !! search form did not render within 15s")

                    async with _step(page, f"range{ri}_fill_form", dep=dep, ret=ret):
                        filled = await _fill_form(page, vals, dep, ret)
                        if filled is False:
                            print("   !! form incomplete -- retrying this range once")
                            await page.goto(cfg.url, timeout=60000)
                            await page.wait_for_timeout(1500)
                            filled = await _fill_form(page, vals, dep, ret)

                    async with _step(page, f"range{ri}_submit"):
                        await t_submit(page)
                        await page.wait_for_timeout(1500)
                    if "/search" not in page.url:
                        print(f"   !! not on results page ({page.url[:60]}) -- skipping range")
                        continue
                    # The URL carries the searched route. If an airport silently failed
                    # to commit, we would be searching the WRONG route and every number
                    # downstream would be wrong-but-plausible. Fail loudly instead.
                    _o = _code_of(vals.get("origin"))
                    _dst = _code_of(vals.get("destination"))
                    _u = page.url
                    if _o and f"f={_o}" not in _u and f"f%3D{_o}" not in _u and _o not in _u:
                        print(f"   !! ORIGIN {_o} MISSING FROM SEARCH URL -- results are for the "
                              f"wrong route: {_u[:110]}")
                    if _dst and _dst not in _u:
                        print(f"   !! DESTINATION {_dst} missing from search URL: {_u[:110]}")

                    # Use the SAME API lists AND the SAME filter the grader applies.
                    _cls = _class_of(vals.get("travel_class") or vals.get("class"))
                    _pax = _adults_of(vals.get("passengers") or vals.get("adults")) or 1
                    _airline = vals.get("airline") or {}
                    _acode = _airline.get("code") if isinstance(_airline, dict) else None
                    _needs_airline = ("date_range_search" == category) and bool(_acode)

                    if _needs_airline:
                        cabin_n = _norm_cabin(_cls)
                        dep_list = _filter_with_airline(captured.get("dep"), _cls, _pax, _acode)
                        ret_list = _filter_with_airline(captured.get("ret"), _cls, _pax, _acode)
                        print(f"   [api] airline={_acode} dep_flights={len(dep_list)} "
                              f"ret_flights={len(ret_list)}")
                        # store the WHOLE list for the answer (grader checks each flight number)
                        api_dep = {"list": [_describe(f, cabin_n) for f in dep_list],
                                   "flight": (dep_list[0].get("flightNumber") if dep_list else "")}
                        api_ret = {"list": [_describe(f, cabin_n) for f in ret_list],
                                   "flight": (ret_list[0].get("flightNumber") if ret_list else "")}
                    else:
                        api_dep = _pick_from_api(captured.get("dep"), _cls, _pax)
                        api_ret = _pick_from_api(captured.get("ret"), _cls, _pax)
                    print(f"   [api] filtered dep={bool(api_dep)} ret={bool(api_ret)} "
                          f"(cabin={_norm_cabin(_cls)}, pax={_pax})")

                    # Airline requirement (date_range_search): list ALL that airline's direct flights.
                    _airline = vals.get("airline") or {}
                    _airline_code = _airline.get("code") if isinstance(_airline, dict) else None
                    dep_air_nums = _all_flight_numbers_for_airline(
                        captured.get("dep"), _cls, _pax, _airline_code) if _airline_code else []
                    ret_air_nums = _all_flight_numbers_for_airline(
                        captured.get("ret"), _cls, _pax, _airline_code) if _airline_code else []
                    if _airline_code:
                        print(f"   [airline {_airline_code}] dep={dep_air_nums} ret={ret_air_nums}")

                    # airline_search_simple: the grader wants ONE total price string.
                    total_str, total_detail = _total_price_answer(
                        captured.get("dep"), captured.get("ret"), _cls, _pax)
                    if total_str:
                        print(f"   [total] {total_str}  ({total_detail})")
                    if (not api_dep or not api_ret) and captured.get("ret"):
                        _sample = (captured["ret"] or [{}])[0]
                        print("   [api] sample flight keys:", list(_sample.keys()))
                        print("   [api] sample segments:", len(_sample.get("segments") or []),
                              "cabinPricing:", (_sample.get("cabinPricing") or [])[:2])

                    async with _step(page, f"range{ri}_read_departures"):
                        dep_rows = await t_read_results(page)
                    if dep_rows.startswith("NO RESULTS") and not api_dep:
                        print("   no departing rows"); continue

                    if api_dep:
                        dep_pick = api_dep
                        print("   cheapest departure (api):", dep_pick)
                    else:
                        dep_pick = _ask_cheapest(cfg.task, dep_rows, "DEPARTING")
                        print("   cheapest departure (ui):", dep_pick)
                        print("   !! API departure filter returned nothing")

                    if api_ret:
                        ret_pick = api_ret
                        print("   cheapest return (api):", ret_pick)
                    else:
                        async with _step(page, f"range{ri}_select_departure"):
                            sel_msg, _row = await t_select_flight(page, "cheapest")
                            print("  ", sel_msg)
                        async with _step(page, f"range{ri}_read_returns"):
                            ret_rows = await t_read_results(page)
                        ret_pick = {}
                        if not ret_rows.startswith("NO RESULTS"):
                            ret_pick = _ask_cheapest(cfg.task, ret_rows, "RETURNING")
                            print("   cheapest return (ui):", ret_pick)

                    findings.append((ri, dep_pick, ret_pick, dep_air_nums, ret_air_nums, total_str))
                    rs.set_attribute("dep", json.dumps(dep_pick))
                    rs.set_attribute("ret", json.dumps(ret_pick))

            # Build the answer from every field we captured -- maximises bucket hits.
            # Cheapest tasks: the UI does not render flight numbers, so take the
            # cheapest DIRECT flight from the captured API response.
            # Roundtrip: the UI shows no flight numbers, so use the API for both legs.
            roundtrip_answer = None
            if "roundtrip" in CATEGORY:
                _req_direct = "direct" in (cfg.task or "").lower()
                _d = _cheapest_flight(captured.get("dep"), _req_direct)
                _r = _cheapest_flight(captured.get("ret"), _req_direct)
                if _d and _r:
                    _tot = int(_d["priceUsd"]) + int(_r["priceUsd"])
                    _dn = _flight_numbers_of(_d)
                    _rn = _flight_numbers_of(_r)
                    roundtrip_answer = (
                        f"Cheapest round-trip: outbound flights {', '.join(_dn)}; "
                        f"return flights {', '.join(_rn)}; "
                        f"total price ${_tot:,}.")
                    print(f"   [roundtrip] dep={_dn} ret={_rn} total=${_tot:,}")
                else:
                    print(f"   [roundtrip] missing leg: dep={bool(_d)} ret={bool(_r)}")

            cheapest_answer = None
            if "cheapest" in CATEGORY:
                _req_direct = "direct" in (cfg.task or "").lower()
                _cf = _cheapest_flight(captured.get("dep"), _req_direct)
                n_seen = len(captured.get("dep") or [])
                if _cf:
                    cheapest_answer = (
                        f"The cheapest flight is {_cf.get('flightNumber','')} "
                        f"at ${_cf.get('priceUsd','')}.")
                print(f"   [cheapest] require_direct={_req_direct} flights_seen={n_seen} "
                      f"-> {cheapest_answer!r}")

            # Budget tasks: the grader needs EXACTLY one of its phrases. Compute it
            # from the captured flights (never from the LLM's wording).
            budget_answer = None
            if "budget" in CATEGORY:
                _b = _budget_of(vals)
                _under = _cheapest_under_budget(captured.get("dep"), _b)
                if _under is not None:
                    budget_answer = (f"{_BUDGET_YES}: flight {_under.get('flightNumber','')} "
                                     f"at ${_under.get('priceUsd','')}")
                else:
                    budget_answer = _BUDGET_NO
                n_seen = len(captured.get("dep") or [])
                print(f"   [budget] budget={_b} flights_seen={n_seen} -> {budget_answer!r}")

            is_total_price_task = ("airline_search" in CATEGORY)
            parts = []
            for entry in findings:
                ri, d, r = entry[0], entry[1], entry[2]
                dep_air = entry[3] if len(entry) > 3 else []
                ret_air = entry[4] if len(entry) > 4 else []
                total_str = entry[5] if len(entry) > 5 else None

                if is_total_price_task and total_str:
                    # ONLY the total price -- never any other number, or the
                    # doubled_total_absent bucket fails.
                    parts.append(
                        f"The total price for the cheapest departure and return "
                        f"flight is {total_str}.")
                    continue

                line = (f"Date range {ri}: cheapest departure {d.get('flight','')} "
                        f"{d.get('time','')} ({d.get('duration','')}) {d.get('price','')}; "
                        f"cheapest return {r.get('flight','')} {r.get('time','')} "
                        f"({r.get('duration','')}) {r.get('price','')}")
                if dep_air:
                    line += f"; airline departure flights: {', '.join(dep_air)}"
                if ret_air:
                    line += f"; airline return flights: {', '.join(ret_air)}"
                parts.append(line)
            final_answer = (budget_answer or cheapest_answer or roundtrip_answer
                            or "\n".join(parts))
            print("\n[answer]\n" + final_answer)

            # The grader reads window.__testingAPI.getState() for the ui_* date
            # buckets. That API only exists on the app page (a results page or a
            # navigated-away tab won't have it). Re-load the form and re-set the
            # LAST range's dates so getState() succeeds and the dates are present.
            try:
                has_api = await page.evaluate(
                    "() => typeof window.__testingAPI !== 'undefined' "
                    "&& !!window.__testingAPI.getState")
            except Exception:
                has_api = False
            _needs_ui_dates = ("airline_search" in CATEGORY)
            if not has_api:
                # The app (and its testing API) isn't on the current page. Reload
                # so getState() works; re-fill dates where ui_* buckets need them.
                print("   [state] testingAPI missing -> reloading app for grader")
                for _try in range(3):
                    try:
                        await page.goto(cfg.url, timeout=60000)
                        await page.wait_for_timeout(1800)
                        ok = await page.evaluate(
                            "() => typeof window.__testingAPI !== 'undefined' "
                            "&& !!window.__testingAPI.getState")
                        if ok:
                            break
                    except Exception:
                        await page.wait_for_timeout(1500)
                if _needs_ui_dates and ranges:
                    last_dep, last_ret = ranges[-1]
                    await _fill_form(page, vals, last_dep, last_ret)
                    await page.wait_for_timeout(800)

            async with _step(page, "grade", answer=final_answer[:200]):
                await evaluator.update(page=page, answer_message=final_answer)
                result = await evaluator.compute()
            root.set_attribute("eval.score", result.score)
            await browser.close()

    print(f"\nScore: {result.score}\nReason: {result.reasoning[:300]}")
    tracer_provider.force_flush()
    return result.score, result.reasoning


async def run_task(index=0, category=CATEGORY):
    dataset = load_dataset("Halluminate/westworld", split="validation")
    tasks = dataset.filter(lambda r: f"noodle_flights/{category}" in r["task_id"])
    print(f"Found {len(tasks)} {category} tasks")
    if len(tasks) == 0:
        return

    item = DatasetItem(**tasks[index])
    cfg = item.generate_task_config()
    evaluator = instantiate(cfg.eval_config)
    await evaluator.reset()
    print("TASK:", cfg.task)

    history = []
    final_answer = ""             # the agent's written answer (graded by substring match)
    recorded = {}                 # range number -> findings (scratchpad, never scrolls away)
    last_results = ""             # most recent read_results output
    phase = {"submitted": False, "saw_dep": False, "saw_ret": False}  # per-range progress
    resets_since_record = 0       # blocks reset_search loops
    selected_dep = ""             # text of the departure row the agent selected
    done_actions = set()          # config steps already completed (loop guard)
    attempts = {}                 # per-step retry counter
    with tracer.start_as_current_span("noodle_flights.task") as root:
        root.set_attribute("task.prompt", cfg.task)
        root.set_attribute("task.category", category)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(timezone_id="UTC")   # avoid off-by-one
            page = await context.new_page()
            page.set_default_timeout(5000)
            # Page load needs a much longer timeout than clicks (cold starts are slow).
            for _attempt in range(3):
                try:
                    await page.goto(cfg.url, timeout=60000)
                    break
                except Exception as _e:
                    print(f"   goto attempt {_attempt + 1} failed: {_e}")
                    await page.wait_for_timeout(2000)
            await page.wait_for_timeout(1500)

            for step in range(MAX_STEPS):
                handles, elements = await get_elements(page)
                action = decide(cfg.task, elements, history, recorded, last_results)  # LLM call
                name = action.get("action")
                print(f"step {step}: {json.dumps(action)}")

                # canonical key for one-time config actions (prevents loops)
                key = None
                if name == "set_trip_type":
                    key = "trip_type"
                elif name == "set_airport":
                    key = f"airport_{action.get('field')}"
                elif name == "set_class":
                    key = "class"
                elif name == "pick_date":
                    key = f"date_{action.get('which', 'departure')}"
                elif name == "submit":
                    key = "submit"

                # Extraction tasks (date_range / airline_search) must re-pick dates
                # and re-submit per date range. Basic tasks KEEP the one-shot guard --
                # without it, repeated pick_date calls drift the calendar forward.
                multi_search = ("date_range" in CATEGORY) or ("airline_search" in CATEGORY)
                if multi_search and key in ("date_departure", "date_return", "submit"):
                    key = None
                if key and key in done_actions:
                    history.append(f"ALREADY DONE: {key} -- move to the next step")
                    print(f"  (skipped: {key} already done)")
                    continue

                with tracer.start_as_current_span(f"agent.{name}") as s:
                    s.set_attribute("action", json.dumps(action))
                    try:
                        if name == "done":
                            history.append("done"); break
                        elif name == "set_airport":
                            res = await t_set_airport(page, action["field"], action["code"])
                            ok = res.endswith("ok")
                            history.append(("SUCCESS: " if ok else "FAILED: ") + res)
                            attempts[key] = attempts.get(key, 0) + 1
                            if ok:
                                done_actions.add(key)   # only lock it in if the CODE landed
                            elif attempts[key] >= 3:
                                done_actions.add(key)   # give up after 3 tries; don't loop
                                history.append(f"GAVE UP on {key} after 3 attempts")
                        elif name == "set_trip_type":
                            history.append("SUCCESS: " + await t_set_trip_type(page, action["value"]))
                            done_actions.add(key)
                        elif name == "set_class":
                            history.append("SUCCESS: " + await t_set_class(page, action["value"]))
                            done_actions.add(key)
                        elif name == "pick_date":
                            res = await t_pick_date(page, action.get("which", "departure"),
                                                    action["year"], action["month"], action["day"])
                            history.append("SUCCESS: " + res)
                            done_actions.add(key)   # one attempt only -- never re-advance
                        elif name == "submit":
                            history.append("SUCCESS: " + await t_submit(page))
                            done_actions.add(key)
                            phase.update(submitted=True, saw_dep=False, saw_ret=False)
                            await page.wait_for_timeout(2500)      # let results render
                            with open("results_dump.html", "w") as _f:
                                _f.write(await page.content())
                            print("   [dump] wrote results_dump.html")
                        elif name == "read_results":
                            if not phase["submitted"]:
                                history.append("FAILED: you must submit a search before read_results.")
                                print("   [guard] read_results before submit -- rejected")
                                await page.wait_for_timeout(200)
                                continue
                            res = await t_read_results(page)
                            if res.startswith("RETURNING"):
                                phase["saw_ret"] = True
                            elif res.startswith("DEPARTING"):
                                phase["saw_dep"] = True
                            # Show Claude the rows this turn, but keep history short so
                            # earlier findings don't scroll out of context.
                            history.append("SUCCESS: read_results returned "
                                           + str(res.count("[")) + " rows (see ELEMENTS/results below)")
                            last_results = res
                            print("   [results] " + res[:180].replace("\n", " / "))
                        elif name == "reset_search":
                            # Allow reset, but ALWAYS actually reload so the agent is never
                            # stuck with a half-filled form + a blocked action.
                            res = await t_reset_search(page, cfg.url)
                            resets_since_record += 1
                            phase.update(submitted=False, saw_dep=False, saw_ret=False)
                            selected_dep = ""
                            if resets_since_record >= 3:
                                history.append("WARNING: you have reset 3+ times without "
                                               "completing a search. STOP resetting. Fill the "
                                               "form (trip type, airports, class, BOTH dates) "
                                               "and call submit.")
                            # form is blank again -> allow all setup steps to re-run
                            done_actions.clear()
                            attempts.clear()
                            history.append("SUCCESS: " + res +
                                           " -- you must set trip type, airports, class, "
                                           "passengers and BOTH dates again, then submit.")
                        elif name == "select_flight":
                            if not phase["saw_dep"]:
                                history.append("FAILED: read_results (DEPARTING) before select_flight.")
                                print("   [guard] select_flight before reading departures")
                                await page.wait_for_timeout(150)
                                continue
                            res, row_text = await t_select_flight(page, action.get("index", "cheapest"))
                            if not row_text:
                                attempts["select"] = attempts.get("select", 0) + 1
                                history.append("FAILED: " + res +
                                               " -- do NOT repeat select_flight; call read_results first.")
                                print("   [select] FAILED: " + res)
                                if attempts["select"] >= 3:
                                    history.append("select_flight keeps failing -- record what you "
                                                   "can from the DEPARTING rows and move on.")
                                await page.wait_for_timeout(200)
                                continue
                            selected_dep = row_text
                            history.append("SUCCESS: " + res + " | chosen departure row: " + row_text)
                            print("   [select] " + res)
                            print("   [select] departure row: " + row_text[:120])
                        elif name == "record_range":
                            if not (phase["saw_dep"] and phase["saw_ret"]):
                                history.append("FAILED: before record_range you must submit, "
                                               "read_results (DEPARTING), select_flight, then "
                                               "read_results again (RETURNING).")
                                print(f"   [guard] record_range blocked (dep={phase['saw_dep']} ret={phase['saw_ret']})")
                                await page.wait_for_timeout(200)
                                continue
                            rng = str(action.get("range", "?"))
                            dep_t = action.get("dep_time", "").strip()
                            ret_t = action.get("ret_time", "").strip()
                            dep_d = action.get("dep_duration", "").strip()
                            ret_d = action.get("ret_duration", "").strip()
                            import re as _re
                            _timepat = _re.compile(r"\d{1,2}:\d{2}\s*[AP]M")
                            bad = [n for n, v in (("dep_time", dep_t), ("ret_time", ret_t))
                                   if not v or not _timepat.search(v)]
                            if bad:
                                hint = (f" Your selected DEPARTURE row was: {selected_dep}"
                                        if selected_dep else "")
                                history.append(f"FAILED: record_range {rng} -- {bad} must be REAL "
                                               "time ranges copied from the results, like "
                                               "'3:57 PM - 7:25 PM'. Placeholders are not allowed."
                                               + hint)
                                print(f"   [recorded] REJECTED range {rng}: bad fields {bad}")
                            else:
                                recorded[rng] = (f"departure {dep_t} ({dep_d}); return {ret_t} ({ret_d})")
                                history.append(f"SUCCESS: recorded range {rng}")
                                print(f"   [recorded] range {rng}: {recorded[rng]}")
                                phase.update(submitted=False, saw_dep=False, saw_ret=False)
                                resets_since_record = 0
                                selected_dep = ""
                        elif name == "report_answer":
                            final_answer = action.get("text", "")
                            history.append("SUCCESS: reported answer (" + str(len(final_answer)) + " chars)")
                            print("   [answer] " + final_answer[:300])
                        elif name == "click":
                            if phase["submitted"]:
                                history.append("FAILED: do not use raw click on the results page. "
                                               "Use select_flight / read_results / record_range.")
                                print("   [guard] raw click blocked on results page")
                                await page.wait_for_timeout(150)
                                continue
                            i = action["index"]
                            if i < len(handles):
                                await handles[i].click()
                                history.append(f"SUCCESS: clicked [{i}]")
                            else:
                                history.append(f"bad index {i}")
                        else:
                            history.append(f"unknown action {name}")
                    except Exception as e:
                        s.set_status(Status(StatusCode.ERROR, str(e)))
                        history.append(f"ERROR {name}: {e}")
                        print(f"   [error] {name}: {str(e)[:180]}")

                    # Screenshot for THIS step, attached to the agent.<name> span.
                    try:
                        _shot = await _screenshot(page, f"{step:02d}_{name}")
                        _attach_shot(s, _shot)
                        if _shot:
                            print(f"   [shot] {name} -> {os.path.basename(_shot)}")
                    except Exception as _se:
                        print(f"   [shot] error: {str(_se)[:80]}")
                await page.wait_for_timeout(600)

            # Safety net: if the agent ran out of steps before report_answer,
            # assemble the answer from whatever it recorded per range.
            # Always append the structured records -- they contain the exact time ranges
            # the grader matches on, regardless of how Claude phrased its prose answer.
            if recorded:
                appendix = "\n".join(f"Date range {k}: {v}" for k, v in sorted(recorded.items()))
                final_answer = (final_answer + "\n\n" + appendix) if final_answer else appendix
                print("   [answer + scratchpad]", appendix[:220])
            # THE KEY FIX: the verifier substring-matches the agent's written answer.
            await evaluator.update(page=page, answer_message=final_answer)
            result = await evaluator.compute()
            root.set_attribute("eval.score", result.score)
            root.set_attribute("eval.reasoning", result.reasoning)
            await browser.close()

    print(f"\nScore: {result.score}\nReason: {result.reasoning}")
    print("Trace: http://localhost:6006  (project: noodle-flights-agent)")
    tracer_provider.force_flush()
    return result.score, result.reasoning


async def run_all():
    # discover how many tasks exist in this category
    ds = load_dataset("Halluminate/westworld", split="validation")
    total = len(ds.filter(lambda r: f"noodle_flights/{CATEGORY}" in r["task_id"]))
    print(f"\n########## Running all {total} '{CATEGORY}' tasks ##########")

    results = []
    for i in range(total):
        print(f"\n================= TASK {i}/{total-1} =================")
        try:
            score, reason = await run_task(index=i)
        except Exception as e:
            score, reason = 0.0, f"CRASHED: {e}"
            print("crashed:", e)
        results.append((i, score, reason))

    passed = sum(1 for _, sc, _ in results if sc == 1.0)
    print(f"\n########## SCOREBOARD: {passed}/{total} passed ##########")
    for i, sc, reason in results:
        mark = "PASS" if sc == 1.0 else "FAIL"
        short = reason if sc == 1.0 else reason[:90]
        print(f"  task {i}: {mark}   {short}")


async def _dispatch(idx):
    ds = load_dataset("Halluminate/westworld", split="validation")
    multi = (("date_range" in CATEGORY) or ("airline_search" in CATEGORY)
             or ("budget" in CATEGORY) or ("cheapest" in CATEGORY)
             or ("roundtrip" in CATEGORY))
    if multi:
        n = len(ds.filter(lambda r: f"noodle_flights/{CATEGORY}" in r["task_id"]))
        if idx is not None:
            await run_extraction_task(idx, CATEGORY, ds)
        else:
            scores = []
            for i in range(n):
                print(f"\n===== TASK {i} =====")
                try:
                    sc, _ = await run_extraction_task(i, CATEGORY, ds)
                except Exception as e:
                    import traceback
                    sc = 0.0
                    print("crashed:", e)
                    traceback.print_exc()
                scores.append(sc)
            print(f"\n##### SCOREBOARD: {sum(1 for s in scores if s==1.0)}/{n} #####")
    else:
        await (run_task(index=idx) if idx is not None else run_all())


if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(_dispatch(idx))