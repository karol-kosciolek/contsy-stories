# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The agent's tools for steering its own run: reading the brief again, the prompt-writing rules of the studio, telling the
owner where the work stands, asking the owner something, and finishing."""
from __future__ import annotations

from . import agent_brief, llm, styles
from .agent_host import Host, ToolError
from .agent_prompt import PHASES
from .agent_tools import choice, many, obj, text, tool, whole

MAX_QUESTIONS = 3


@tool("get_brief", "The brief the owner attached, again: the list of files (with the ids of pictures) and the text of the "
                   "text and Word files, marked as client material. PDFs are in the first message only.",
      obj({"file": whole("Only this file (1 is the first).", 1, 30)}))
def _get_brief(host: Host, args: dict):
    files = host.brief_files()
    wanted = args.get("file")
    rows, texts = [], []
    for number, item in enumerate(files, 1):
        row = {"file": number, "name": item.get("name", ""), "kind": item.get("kind", "")}
        if item.get("kind") == "image":
            row["picture_id"] = str(item.get("ref", ""))[:12]
        rows.append(row)
        if wanted not in (None, number) or item.get("kind") not in ("text", "docx"):
            continue
        path = host.store.path("brief", f"{item.get('sha256')}.{item.get('suffix')}")
        try:
            data = path.read_bytes()
            body = (agent_brief.docx_text if item["kind"] == "docx" else agent_brief.decode_text)(data, item.get("name", ""))
        except (OSError, ValueError, agent_brief.BriefError):
            body = "(this file could not be read again)"
        texts.append(f"[[client-material {host.nonce} · file {number} · {item.get('name', '')}]]\n{body}\n"
                     f"[[end client-material {host.nonce}]]")
    extra = host.store.read().get("extra") or ""
    return {"files": rows, **({"owner_instructions": extra} if extra else {}), "texts": texts}


AGENT_NOTE = ("\n\nNote for the agent: the advice above to put text seen in the picture in quotes is for text that does not "
              "have to be exact (a sign in the background that nobody reads). Text that must be right (an end card, a "
              "tagline, an address, a price, a product name) is never drawn by the model: set it with set_overlays.")


@tool("get_prompt_rules", "How the studio writes scene and picture descriptions for its video and image models (the "
                          "instructions of its own prompt writer). Read it once before writing scenes.",
      obj({"kind": choice("video for scenes and clips, image for pictures.", "video", "image"),
           "style": text("A style id from list_styles (default Realistic).", max_length=200)}, ("kind",)))
def _get_prompt_rules(host: Host, args: dict):
    style = args.get("style") or styles.REALISTIC
    if not styles.valid(style):
        raise ToolError("INVALID", "style is not a style of the studio (list_styles).")
    return llm.system_prompt(args["kind"], style) + AGENT_NOTE


@tool("update_status", "Tell the owner where the work stands: shown on the Agent page while the run works. Call it as each "
                       "phase starts.",
      obj({"phase": choice("The phase.", *PHASES), "text": text("One short sentence of what you are doing now.",
                                                                 max_length=300, min_length=1)}, ("phase", "text")))
def _update_status(host: Host, args: dict):
    host.store.update(status={"phase": args["phase"], "text": args["text"]})
    host.store.log("status", phase=args["phase"], text=args["text"])
    return "Noted."


@tool("ask_owner", "Record a question the owner should decide (a missing logo, a doubtful likeness, a choice between "
                   "two readings of the brief). The run does not wait: carry on with the safest assumption and report it. "
                   f"At most {MAX_QUESTIONS} questions.",
      obj({"question": text("The question, with what you assume meanwhile.", max_length=600, min_length=1)}, ("question",)))
def _ask_owner(host: Host, args: dict):
    questions = list(host.store.read().get("questions") or [])
    if len(questions) >= MAX_QUESTIONS:
        raise ToolError("INVALID", f"You have asked {MAX_QUESTIONS} times already. Decide yourself and report it in "
                        "finish_run.")
    questions.append(args["question"])
    host.store.update(questions=questions)
    host.store.log("question", text=args["question"])
    return "Recorded. The owner sees it when the run ends. Carry on with the safest assumption."


ITEM = obj({"requirement": text("One requirement or deliverable of the brief.", max_length=300, min_length=1),
            "status": choice("done, partial or skipped.", "done", "partial", "skipped"),
            "note": text("What was done, or why not.", max_length=300)}, ("requirement", "status"))


REVIEW = obj({"scene": whole("The scene number (1 is the first).", 1, 60),
              "verdict": choice("ok, fixed or issue.", "ok", "fixed", "issue"),
              "note": text("What you saw.", max_length=300)}, ("scene", "verdict"))


def _gates(host: Host) -> list:
    """The quality checks of the package (app/agent_delivery.py) that fail now: (code, text) pairs."""
    try:
        from . import agent_delivery
    except ImportError:
        return []
    return agent_delivery.gate_failures(host)


@tool("finish_run",
      "End the run: the only way it ends well. Report honestly what is done and what is not. 'complete' means every item "
      "of the brief is done and the checks pass; with anything open, say 'partial' and list it. The studio checks the "
      "package before it accepts 'complete'.",
      obj({"outcome": choice("complete or partial.", "complete", "partial"),
           "summary": text("What was done, in a few sentences.", max_length=3000, min_length=1),
           "checklist": many("Every requirement of the brief and how it went.", ITEM, max_items=60),
           "assumptions": many("What you assumed.", text("An assumption.", max_length=300), max_items=20),
           "deviations": many("Where the result differs from the brief, and why.", text("A deviation.", max_length=300),
                              max_items=20),
           "owner_should_check": many("What the owner should look at by eye.", text("An item.", max_length=300),
                                      max_items=20),
           "review": many("What you saw of each scene of the finished film, compared with the brief: ok, fixed (you corrected "
                          "it) or issue (still not right).", REVIEW, max_items=60)},
          ("outcome", "summary", "checklist")))
def _finish_run(host: Host, args: dict):
    open_items = [row for row in args["checklist"] if row["status"] != "done"]
    quality = [(code, message) for code, message in _gates(host) if code in ("G8", "G9")]
    if args["outcome"] == "partial" and quality and not (args.get("owner_should_check") or args.get("deviations")):
        raise ToolError("REVIEW_REQUIRED", "The shot or sound check did not pass: list every failing shot and sound "
                        "problem in owner_should_check (the owner must see them): "
                        + " ".join(f"[{code}] {message}" for code, message in quality))
    if args["outcome"] == "complete":
        if open_items:
            raise ToolError("REVIEW_REQUIRED", f"{len(open_items)} checklist item(s) are not done. Finish them, or "
                            "finish with outcome partial and say what is open.")
        if failing := _gates(host):
            raise ToolError("REVIEW_REQUIRED", "The package is not ready: " + " ".join(f"[{code}] {message}"
                                                                                       for code, message in failing))
    record = {key: args.get(key, []) for key in ("checklist", "assumptions", "deviations", "owner_should_check", "review")}
    record["quality"] = [{"code": code, "text": message} for code, message in quality]  # the binding checks
    record["saw_scenes"] = [number + 1 for number in host.store.read().get("saw") or []]  # numbered from 1, as the owner reads
    record.update(kind="complete" if args["outcome"] == "complete" else "open_items", summary=args["summary"],
                  failing=[{"code": code, "text": message} for code, message in _gates(host)])
    host.finished = record
    host.store.update(outcome=record)
    host.store.log("finish", outcome=record["kind"])
    return "The run is finishing. Nothing more to do."
